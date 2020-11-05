from models.ops.complex_ops import *
from models.layers.complex_layers import *
from models.layers.kernelized_layers import *
from tensorflow.keras.layers import Dense, Activation, LSTM, Input, Lambda
import tensorflow as tf
from complexnn import GetImag, GetReal,CReLU,CRot

def log10(x):

    return tf.math.log(x) / 2.302585092994046

class CNBF(tf.keras.Model):
    def __init__(self,
                 config,
                 fgen,
                 batch_size,
                 n_ch_base=16,
                 kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                 kernel_initializer=tf.keras.initializers.he_normal(),
                 name = "cnbf",
                 dropout=0.0):
        super(CNBF, self).__init__()
        self.dropout = dropout
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

    def build(self,input_shape):
        self.layer_0 = Lambda(self.layer0)
        self.complex_dense_1 = Complex_Dense(units=50, activation='none')
        self.crot_1 = CRot()
        self.complex_dense_2 = Complex_Dense(units=self.nbin, activation='none')
        self.crot_2 = CRot()
        self.layer_1 = Lambda(self.layer1)
        self.complex_lstm = Kernelized_Complex_LSTM(units=self.nmic * 2)
        self.complex_dense_3 = Kernelized_Complex_Dense(units=self.nmic * 2, activation='none')
        self.crot_3 = CRot()
        self.complex_dense_4 = Kernelized_Complex_Dense(units=self.nmic, activation='linear')
        self.layer_2 = Lambda(self.layer2)

    def to_real_imag_tensor(self,x):
        return tf.concat([tf.math.real(x),tf.math.imag(x)],axis=-1)

    def to_complex_tensor(self,x):
        return tf.complex(x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:])

    def call(self,Fs,Fn,training=False):
        Fz = Fs + Fn
        X, Y = self.layer_0(Fz)
        Y = self.complex_dense_1(Y)  # shape = (nbatch, nfram, 50)
        Y = self.crot_1(self.to_real_imag_tensor(Y))
        Y = self.to_complex_tensor(Y)
        Y = self.complex_dense_2(Y)  # shape = (nbatch, nfram, nbin)
        Y = self.crot_2(self.to_real_imag_tensor(Y))
        Y = self.to_complex_tensor(Y)
        X = self.layer_1([X, Y])
        X = self.complex_lstm(X)  # shape = (nbatch, nfram, nbin, nmic*2)
        X = self.complex_dense_3(X)  # shape = (nbatch, nfram, nbin, nmic*2)
        X = self.crot_3(self.to_real_imag_tensor(X))
        X = self.to_complex_tensor(X)
        W = self.complex_dense_4(X)  # shape = (nbatch, nfram, nbin, nmic)
        Fys, Fyn, cost = self.layer_2([Fs, Fn, W])
        return Fys, Fyn,0.0,cost,cost,0.0

    
