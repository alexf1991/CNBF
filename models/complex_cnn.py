
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D,\
                                                    ComplexConv3D,\
                                                    ComplexDense,\
                                                    FFT,IFFT,FFT2,IFFT2,\
                                                    SpectralPooling1D,SpectralPooling2D,ComplexFRN
from complexnn import GetImag, GetReal,CReLU
from models.ops.complex_ops import *
from models.layers.complex_layers import *
from models.layers.kernelized_layers import *
from tensorflow.keras.layers import Dense, Activation, LSTM, Input, Lambda
import tensorflow as tf


def log10(x):
    return tf.math.log(x) / 2.302585092994046


class CNBF_CNN(tf.keras.Model):
    def __init__(self,
                 config,
                 fgen,
                 batch_size,
                 n_ch_base = 16,
                 kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                 kernel_initializer=tf.keras.initializers.he_normal(),
                 name="cnbf",
                 dropout=0.0):
        super(CNBF_CNN, self).__init__()
        self.dropout = dropout
        self.n_ch_base = n_ch_base
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
        self.convArgs.update({"spectral_parametrization":config["spectral_parametrization"],
                     "kernel_initializer": config["kernel_initializer"]})

    # ---------------------------------------------------------
    def layer0(self, inp):
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
    def layer2(self, inp):
        Fs = tf.cast(inp[0], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)
        Fn = tf.cast(inp[1], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)
        W = tf.cast(inp[2], tf.complex64)  # shape = (nbatch, nfram, nbin, nmic)

        # beamforming
        W = vector_normalize_magnitude(W)  # shape = (nbatch, nfram, nbin, nmic)
        W = vector_normalize_phase(W)  # shape = (nbatch, nfram, nbin, nmic)
        Fys = vector_conj_inner(Fs, W)  # shape = (nbatch, nfram, nbin)
        Fyn = vector_conj_inner(Fn, W)  # shape = (nbatch, nfram, nbin)

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

        cost = -tf.reduce_mean(delta_snr, axis=(1, 2))

        return [Fys, Fyn, cost]

    def build(self, input_shape):
        self.layer_0 = Lambda(self.layer0)
        self.conv_0 = ComplexConv2D(self.n_ch_base, (3,3), name='conv_0', **self.convArgs)
        self.bn_0 = ComplexBN(name='bn_0', **self.bnArgs)
        self.crelu_0 = CReLU()

        self.conv_1 = ComplexConv2D(self.n_ch_base*2, (3, 3), name='conv_1', **self.convArgs)
        self.bn_1 = ComplexBN(name='bn_1', **self.bnArgs)
        self.crelu_1 = CReLU()

        self.conv_2 = ComplexConv2D(self.n_ch_base*4, (3, 3), name='conv_2', **self.convArgs)
        self.bn_2 = ComplexBN(name='bn_2', **self.bnArgs)
        self.crelu_2 = CReLU()

        self.conv_3 = ComplexConv2D(self.nmic, (3, 3), name='conv_3', **self.convArgs)

        self.layer_2 = Lambda(self.layer2)

    def call(self, Fs, Fn, training=False):
        Fz = Fs + Fn
        X, _ = self.layer_0(Fz)
        X = tf.concat([tf.math.real(X),tf.math.imag(X)],axis=-1)
        X = self.conv_0(X)  # shape = (nbatch, nfram, n_bin,n_ch_base)
        X = self.bn_0(X,training)  # shape = (nbatch, nfram, nbin)
        X = self.crelu_0(X)

        X = self.conv_1(X)  # shape = (nbatch, nfram, n_bin,2*n_ch_base)
        X = self.bn_1(X,training)  # shape = (nbatch, nfram, nbin)
        X = self.crelu_1(X)

        X = self.conv_2(X)  # shape = (nbatch, nfram, n_bin,4*n_ch_base)
        X = self.bn_2(X,training)  # shape = (nbatch, nfram, nbin)
        X = self.crelu_2(X)

        W = self.conv_3(X)
        W = tf.complex(W[...,:W.shape[-1]//2],W[...,W.shape[-1]//2:]) # shape = (nbatch, nfram, n_bin,nmic)

        Fys, Fyn, cost = self.layer_2([Fs, Fn, W])
        return Fys, Fyn, cost

