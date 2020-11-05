
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D,\
                                                    ComplexConv3D,\
                                                    ComplexDense,\
                                                    FFT,IFFT,FFT2,IFFT2,\
                                                    SpectralPooling1D,SpectralPooling2D,ComplexFRN
from complexnn import GetImag, GetReal,CReLU,CRot,Cauchy1
from models.ops.complex_ops import *
from models.layers.complex_layers import *
from models.layers.kernelized_layers import *
from models.layers.spectral_normalization import SpectralNormalization
from tensorflow.keras.layers import Dense, Activation, LSTM, Input, Lambda
import tensorflow as tf


def log10(x):
    return tf.math.log(x) / 2.302585092994046


class CriticLayer(tf.keras.layers.Layer):
    def __init__(self, ch_base, use_bias=False,
                 use_sigmoid=False, kernel_regularizer=None, kernel_initializer=None,
                 activation=tf.keras.layers.ReLU, name='', use_spectral_normalization=True):
        super(CriticLayer, self).__init__()
        self.ch_base = ch_base
        self.use_sigmoid = use_sigmoid
        self.regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.name_op = name
        self.use_bias_var = use_bias
        self.activation = activation
        self.use_spectral_normalization = use_spectral_normalization

    def build(self, input_shape):

        ct = 0
        run = True
        is_last_conv = False
        self.layers = []

        current_shape = int(np.sqrt(input_shape[-2]))

        while run:

            if is_last_conv:
                conv = tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    activation=None,
                    use_bias=False,
                    name=self.name_op + '_critic_conv_' + str(ct),
                    kernel_regularizer=None,
                    padding="same")
                if self.use_spectral_normalization:
                    conv = SpectralNormalization(conv)

                self.layers.append((conv, 0))
            else:
                conv = tf.keras.layers.Conv2D(
                    filters=(ct + 1) * self.ch_base,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    activation=None,
                    use_bias=False,
                    name=self.name_op + '_critic_conv_' + str(ct),
                    kernel_regularizer=None,
                    padding="same")

                if self.use_spectral_normalization:
                    conv = SpectralNormalization(conv)

                self.layers.append((conv, 0))
                self.layers.append((tf.keras.layers.BatchNormalization(axis=-1,
                                                                       scale=False,
                                                                       name=self.name_op + "_bn_" + str(ct),
                                                                       center=False), 1))
                self.layers.append((self.activation(), 0))

            ct += 1

            current_shape = tf.math.ceil(current_shape / 2)

            if current_shape == 1:
                run = False
                if self.use_sigmoid:
                    self.layers.append((tf.keras.layers.Activation("sigmoid"), 0))

            elif current_shape == 2:

                is_last_conv = True



    def call(self, input, training=False):

        x = input
        for layer in self.layers:

            if layer[1] == 0:
                x = layer[0](x)
            else:
                x = layer[0](x, training)

        return x

class ComplexNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(ComplexNormalization, self).__init__()

    def to_real_imag_tensor(self, x):
        return tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)

    def to_complex_tensor(self, x):
        return tf.complex(x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])

    def call(self,input):
        C = self.to_complex_tensor(input)
        #C = vector_normalize_magnitude(C)
        C = vector_normalize_phase(C)
        return self.to_real_imag_tensor(C)


class CompoundLayer(tf.keras.layers.Layer):
    def __init__(self,n_ch,bnArgs,convArgs,dilation_rate=(1,1),strides=(1,1)):
        self.n_ch = n_ch
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.bnArgs = bnArgs
        self.convArgs = convArgs
        super(CompoundLayer, self).__init__()

    def build(self,input_shape):

        self.bn = ComplexBN(name='bn', **self.bnArgs)
        self.crelu = ComplexNormalization()
        self.conv = ComplexConv2D(self.n_ch, (3, 3), **self.convArgs,strides=self.strides,dilation_rate=self.dilation_rate)


    def call(self,input,training=False):

        X = self.bn(input,training)  # shape = (nbatch, nfram, nbin)
        X = self.crelu(X)
        X = self.conv(X)  # shape = (nbatch, nfram, n_bin,2*n_ch_base)
        return X

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self,n_ch,bnArgs,convArgs,dilation_rate=(1,1),strides=(1,1)):
        self.n_ch = n_ch
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.bnArgs = bnArgs
        self.convArgs = convArgs
        super(DenseLayer, self).__init__()

    def complex_concat(self,x,y):
        x_real = x[...,:x.shape[-1]//2]
        x_img = x[..., x.shape[-1] // 2:]

        y_real = y[...,:y.shape[-1]//2]
        y_img = y[...,y.shape[-1] // 2:]

        return tf.concat([x_real,y_real,x_img,y_img],axis=-1)

    def build(self,input_shape):

        self.bn = ComplexBN(name='bn', **self.bnArgs)
        self.crelu = ComplexNormalization()
        self.conv = ComplexConv2D(self.n_ch, (3, 3), **self.convArgs,strides=self.strides,dilation_rate=self.dilation_rate)


    def call(self,input,training=False):
        Y = input
        X = self.bn(Y, training)  # shape = (nbatch, nfram, nbin)
        X = self.crelu(X)
        X = self.conv(X)  # shape = (nbatch, nfram, n_bin,2*n_ch_base)
        X = self.complex_concat(X,Y)
        return X

class CNBF_CNN(tf.keras.Model):
    def __init__(self,
                 config,
                 fgen,
                 batch_size,
                 n_ch_base = 8,
                 n_dense = 4,
                 kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                 kernel_initializer=tf.keras.initializers.he_normal(),
                 name="cnbf",
                 dropout=0.0):
        super(CNBF_CNN, self).__init__()
        self.dropout = dropout
        self.n_ch_base = n_ch_base
        self.n_dense = n_dense
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.model_name = name
        self.config = config
        self.weights_file = self.config['weights_path'] + name + '.h5'
        self.predictions_file = self.config['predictions_path'] + name + '.mat'
        self.nbatch = batch_size
        self.nfram = fgen.nfram
        self.nbin = fgen.nbin
        self.nmic = fgen.nmic

        self.bnArgs =  {
		"axis":                     -1,
		"momentum":                 0.9,
		"epsilon":                  1e-04
	    }

        self.convArgs = {
            "padding":                  "same",
            "use_bias":                 False,
            "kernel_regularizer":       tf.keras.regularizers.l2(0.0001),
        }
        self.convArgs.update({"spectral_parametrization":bool(config["spectral_parametrization"]),
                     "kernel_initializer": config["kernel_initializer"]})


    def build(self, input_shape):
        self.layer_0 = self.layer0
        self.init_conv = ComplexConv2D(self.n_ch_base, (3,3), name='init_conv', **self.convArgs)
        #self.init_conv2 = ComplexConv2D(self.n_ch_base, (3, 3), name='init_conv2', **self.convArgs)

        self.dense_layers = []
        for i_dense in range(self.n_dense):
            self.dense_layers.append(DenseLayer(self.n_ch_base,
                                                dilation_rate = [2**i_dense,1],
                                                bnArgs=self.bnArgs,
                                                convArgs=self.convArgs))

        self.conv_after_db = ComplexConv2D(self.n_ch_base*self.n_dense, (3, 3), name='conv_after_db', **self.convArgs)
        self.bn_after_db = ComplexBN(name='bn_after_db', **self.bnArgs)
        self.act_after_db = ComplexNormalization()
        #self.conv_after_db2 = ComplexConv2D(self.n_ch_base * self.n_dense, (3, 3), name='conv_after_db2', **self.convArgs)
        #self.bn_after_db2 = ComplexBN(name='bn_after_db2', **self.bnArgs)
        #self.act_after_db2 = ComplexNormalization()
        self.output_conv = ComplexConv2D(self.nmic, (3, 3), name='output_conv', **self.convArgs)

        self.layer_2 = self.layer2

        self.folding_layer_1 = ComplexConv2D(1, (3, 3), name='folding_conv_1', **self.convArgs)
        #self.folding_layer_2 = ComplexConv2D(1, (3, 3), name='folding_conv_2', **self.convArgs)
        #self.critic = CriticLayer(8)

    def to_real_imag_tensor(self,x):
        return tf.concat([tf.math.real(x),tf.math.imag(x)],axis=-1)

    def to_complex_tensor(self,x):
        return tf.complex(x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:])

    # ---------------------------------------------------------
    def layer0(self, inp,training):
        Fz = tf.cast(inp, tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)

        Pz = elementwise_abs2(Fz)  # shape = (nbatch, nfram, nbin, nmic)
        Pz = tf.reduce_mean(Pz, axis=-1)  # shape = (nbatch, nfram, nbin)
        Lz = tf.math.log(Pz + 1e-3)[..., tf.newaxis]  # shape = (nbatch, nfram, nbin, 1)
        Lz = elementwise_complex(Lz)

        vz = vector_normalize_magnitude(Fz)  # shape = (nbatch, nfram, nbin, nmic)
        vz = vector_normalize_phase(vz)  # shape = (nbatch, nfram, nbin, nmic)
        X = tf.concat([vz, Lz], axis=-1)  # shape = (nbatch, nfram, nbin, nmic+1)
        Y = tf.reshape(X,
                       [-1, self.nfram, self.nbin * (self.nmic + 1)])  # shape = (nbatch, nfram, nbin*(nmic+1))

        return [X, Y]

    # ---------------------------------------------------------
    def layer1(self, inp):
        X = tf.cast(inp[0], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic+1)
        Y = tf.cast(inp[1], tf.complex64)  # shape = (nbatch, nfram, nbin)

        X = tf.concat([X, Y[..., tf.newaxis]], axis=-1)

        return X

    # ---------------------------------------------------------
    def layer2(self, inp,training):
        Fs = tf.cast(inp[0], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)
        Fn = tf.cast(inp[1], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)
        W = tf.cast(inp[2], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)
        X = inp[3]
        #W = tf.reduce_mean(W,axis=-2,keepdims=True)
        #W = tf.tile(W,[1,1,self.nbin,1])
        # beamforming
        W = vector_normalize_magnitude(W)  # shape = (nbatch, nfram, nbin, nmic)
        W = vector_normalize_phase(W)  # shape = (nbatch, nfram, nbin, nmic)

        Fys = vector_conj_inner(Fs, W)

        Rys = tf.expand_dims(Fys,-1)
        Rys = self.to_real_imag_tensor(Rys)
        X = self.folding_layer_1(X)
        Rys = Rys*X
        #Rys = self.folding_layer_2(Rys)
        Rys = tf.squeeze(self.to_complex_tensor(Rys))

        #Fys = tf.squeeze(W[...,:W.shape[-1]//2])  # shape = (nbatch, nfram, nbin)
        Fyn = vector_conj_inner(Fn, W)
        #Fyn = tf.squeeze(W[...,W.shape[-1]//2:])# shape = (nbatch, nfram, nbin)

        # energy of the input
        Ps = tf.reduce_mean(elementwise_abs2(Fs), axis=-1)  # input (desired source)
        Pn = tf.reduce_mean(elementwise_abs2(Fn), axis=-1)  # input (unwanted source)
        Ls = 10 * log10(Ps + 1e-2)
        Ln = 10 * log10(Pn + 1e-2)

        # energy of the beamformed outputs
        Pys = elementwise_abs2(Fys)  # output (desired source)
        Pyn = elementwise_abs2(Fyn)  # output (unwanted source)
        Lys = 10 * log10(Pys + 1e-2)
        Lyn = 10 * log10(Pyn + 1e-2)

        delta_snr = Lys - Lyn - (Ls - Ln)

        #delta_frame_1 = W[:,:-1,:,:]
        #delta_phase_1 = 0
        #for i_mic in range(1,W.shape[-1]):
        #    delta_phase_1 += tf.abs(tf.math.angle(delta_frame_1[...,i_mic-1])-tf.math.angle(delta_frame_1[...,i_mic]))
        #delta_frame_2 = W[:, 1:, :, :]
        #delta_phase_2 = 0
        #for i_mic in range(1,W.shape[-1]):
        #    delta_phase_2 += tf.abs(tf.math.angle(delta_frame_2[...,i_mic-1])-tf.math.angle(delta_frame_2[...,i_mic]))
        #delta_phase = tf.reduce_mean(tf.abs(delta_phase_2-delta_phase_1))
        #fake = self.critic(tf.expand_dims(tf.stop_gradient(Pys),-1),training)
        #real = self.critic(tf.expand_dims(tf.stop_gradient(Ps),-1), training)
        #critic_loss = tf.reduce_mean(real)-tf.reduce_mean(fake)
        #gen_loss = tf.reduce_mean(self.critic(tf.expand_dims(Pys,-1), training))
        ws_loss = 0.0#gen_loss+critic_loss
        Ps_normed = Ps/tf.norm(Ps,axis=-1,keepdims=True)
        Pn_normed = Pn / tf.norm(Pn, axis=-1, keepdims=True)
        Pys_normed = Pys / tf.norm(Pys, axis=-1, keepdims=True)
        Pyn_normed = Pyn / tf.norm(Pyn, axis=-1, keepdims=True)

        #opt_loss = -tf.reduce_mean(Ps_normed*Pys_normed)\
        #           -tf.reduce_mean(Pn_normed*Pyn_normed)\
        #           +tf.reduce_mean(Pys_normed*Pyn_normed)\
        #           +tf.reduce_mean(tf.abs(Pys-Ps))\
        #           +tf.reduce_mean(tf.abs(Pyn-Pn))+0.1*ws_loss#-tf.reduce_mean(delta_snr, axis=(1, 2))+1e-2*delta_phase
        cost = -tf.reduce_mean(delta_snr, axis=(1, 2))
        opt_loss = cost+0.1*ws_loss
        return [Fys, Fyn,Rys,cost,opt_loss,ws_loss]

    def complex_concat(self,x,y):
        x_real = x[...,:x.shape[-1]//2]
        x_img = x[..., x.shape[-1] // 2:]

        y_real = y[...,:y.shape[-1]//2]
        y_img = y[...,y.shape[-1] // 2:]

        return tf.concat([x_real,y_real,x_img,y_img],axis=-1)

    def complex_mult(self,x,y):
        x_r = x[...,:x.shape[-1]//2]
        x_i = x[..., x.shape[-1] // 2:]

        y_r = y[...,:y.shape[-1]//2]
        y_i = y[..., y.shape[-1] // 2:]

        out_r = x_r*y_r-x_i*y_i
        out_i = x_i*y_r+x_r*y_i

        return tf.concat([out_r,out_i],axis=-1)

    def call(self, Fs, Fn, training=False):
        Fz = Fs + Fn

        X, _ = self.layer_0(Fz,training)
        X = tf.concat([tf.math.real(X),tf.math.imag(X)],axis=-1)

        X = self.init_conv(X)  # shape = (nbatch, nfram, n_bin,n_ch_base)
        #Y = self.init_conv2(X)
        #X = self.complex_mult(X,Y)

        for i_dense in range(self.n_dense):
            X=self.dense_layers[i_dense](X,training=training)
        #Y = X
        X = self.conv_after_db(X)
        X = self.bn_after_db(X,training)
        X = self.act_after_db(X)

        #Y = self.conv_after_db2(Y)
        #Y = self.bn_after_db2(Y,training)
        #Y = self.act_after_db2(Y)
        #X = self.complex_mult(X,Y)

        W = self.output_conv(X)
        W = self.to_complex_tensor(W) # shape = (nbatch, nfram, n_bin,nmic)

        Fys, Fyn,Rys,cost,opt_loss,ws_loss = self.layer_2([Fs, Fn, W,X],training)
        return Fys, Fyn,Rys, cost,opt_loss,ws_loss

