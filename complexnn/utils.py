#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda
import tensorflow as tf
import numpy as np

#
# GetReal/GetImag Lambda layer Implementation
#




def get_realpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, :input_dim]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_imagpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, input_dim:]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, input_dim:]
    elif ndim == 4:
        return x[:, :, :, input_dim:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:]


def get_abs(x):
    real = get_realpart(x)
    imag = get_imagpart(x)

    return K.sqrt(real * real + imag * imag)


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)


class GetReal(Layer):
    def call(self, inputs):
        return get_realpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetImag(Layer):
    def call(self, inputs):
        return get_imagpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetAbs(Layer):
    def call(self, inputs):
        return get_abs(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)

@tf.custom_gradient
def sign(x):
    y = tf.math.sign(x)
    def grad(dy):
        bw_y = tf.nn.tanh(x)
        grads = tf.gradients(bw_y,x)
        dy = dy*grads[0]
        return dy
    return y, grad

class CSP(tf.keras.layers.Layer):
    def __init__(self):
        super(CSP, self).__init__()

    def build(self, input_shape):
        self.act = tf.keras.layers.Activation("relu")
        self.affine_params = self.add_weight(name='affine_params',
                                             shape=[1] * (len(input_shape)-1)+[input_shape[-1]//2] + [2],
                                             initializer=tf.keras.initializers.Zeros,
                                             regularizer=None,
                                             trainable=True,
                                             dtype=tf.float32,
                                             aggregation=tf.VariableAggregation.MEAN)
        shp = [1] * len(input_shape)
        self.one_i = tf.complex(tf.ones(shp)/tf.sqrt(2.0),tf.ones(shp)/tf.sqrt(2.0))
        init_shp = [1] * (len(input_shape)-1)+[input_shape[-1]//2]+[1]
        initialization = tf.concat([tf.random.normal(init_shp),
                                    tf.random.normal(init_shp)], axis=-1)
        self.affine_params.assign(initialization)

    def call(self,inputs,training=False):

        input_real = inputs[...,:inputs.shape[-1]//2]#self.act(inputs[...,:inputs.shape[-1]//2])
        input_imag = inputs[...,inputs.shape[-1] // 2:]#self.act(inputs[...,inputs.shape[-1] // 2:])
        norm = tf.maximum(tf.sqrt(input_real**2+input_imag**2),1e-8)
        norm_k = tf.sqrt(self.affine_params[...,0]**2+self.affine_params[...,1]**2)
        k_real = self.affine_params[...,0]/norm_k
        k_imag = self.affine_params[...,1]/norm_k
        input_complex = tf.complex(input_real,input_imag)
        k_complex = tf.complex(self.affine_params[...,0]/norm_k,self.affine_params[...,1]/norm_k)
        #fact = input_real/norm*k_real+input_imag/norm*k_imag
        fact = input_complex*k_complex
        #tf.print(tf.math.real(fact))
        out_real = input_real+self.act(tf.math.real(fact))#self.act(input_real-self.act(tf.math.real(fact)))#input_real - (tf.math.real(fact))
        out_imag = input_imag+self.act(tf.math.imag(fact))#self.act(input_imag-self.act(tf.math.real(fact)))#input_imag - (tf.math.imag(fact))
        #out_real = norm*fact*k_real
        #out_imag = norm*fact*k_imag

        return tf.concat([out_real,out_imag],axis=-1)

class CRot(tf.keras.layers.Layer):
    def __init__(self):
        super(CRot, self).__init__()

    def build(self, input_shape):
        self.act = tf.keras.layers.Activation("relu")
        self.affine_params = self.add_weight(name='affine_params',
                                             shape=[1] * (len(input_shape)-1)+[input_shape[-1]//2] + [2],
                                             initializer=tf.keras.initializers.Zeros,
                                             regularizer=None,
                                             trainable=True,
                                             dtype=tf.float32,
                                             aggregation=tf.VariableAggregation.MEAN)
        shp = [1] * len(input_shape)
        self.one_i = tf.complex(tf.ones(shp)/tf.sqrt(2.0),tf.ones(shp)/tf.sqrt(2.0))
        init_shp = [1] * (len(input_shape)-1)+[input_shape[-1]//2]+[1]
        initialization = tf.concat([tf.random.normal(init_shp),
                                    tf.random.normal(init_shp)], axis=-1)
        self.affine_params.assign(initialization)

    def call(self,inputs,training=False):

        input_real = self.act(inputs[...,:inputs.shape[-1]//2])
        input_imag = self.act(inputs[...,inputs.shape[-1] // 2:])
        norm = tf.maximum(tf.sqrt(input_real**2+input_imag**2),1e-8)
        k_real = 1#self.affine_params[...,0]
        k_imag = norm#*self.affine_params[...,1]
        norm_k = tf.sqrt(k_real**2+k_imag**2)
        k_real = k_real/norm_k
        k_imag = k_imag/norm_k
        input_complex = tf.complex(input_real,input_imag)
        k_complex = tf.complex(k_real,k_imag)
        #fact = input_real/norm*k_real+input_imag/norm*k_imag
        fact = input_complex*k_complex
        #tf.print(tf.math.real(fact))
        out_real = tf.math.real(fact)#self.act(input_real-self.act(tf.math.real(fact)))#input_real - (tf.math.real(fact))
        out_imag = tf.math.imag(fact)#self.act(input_imag-self.act(tf.math.real(fact)))#input_imag - (tf.math.imag(fact))
        #out_real = norm*fact*k_real
        #out_imag = norm*fact*k_imag

        return tf.concat([out_real,out_imag],axis=-1)

class DenseHess(tf.keras.layers.Layer):
    def __init__(self,units):
        self.units = units
        super(DenseHess, self).__init__()

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units)

    def call(self,inputs,training=False):

        y = self.dense(inputs)
        H = tf.hessians(y,inputs)
        self.add_loss(tf.reduce_sum(H**2))
        return y

class Spline(Layer):
    def __init__(self, lamb=0.0, K=1):
        self.lamb = lamb
        self.K = K
        super(Spline, self).__init__()

    def build(self, input_shape):
        
        self.n_activations = self.add_weight(name = 'n_activations',shape=[1],trainable=False)
        self.n_activations.assign([tf.cast(tf.reduce_prod(input_shape[1:]),tf.float32)])

        #self.a_param = self.add_weight(name='a_param',
                                           #shape=[1] + [self.K] + [1] * (len(input_shape) - 1),# + [1],
                                           #initializer=tf.keras.initializers.Ones,
                                           #regularizer=None,#tf.keras.regularizers.l2(self.lamb),
                                           #dtype = tf.float32,
                                           #trainable=True,
                                           #aggregation=tf.VariableAggregation.MEAN)
        
        #self.b_param = self.add_weight(name='b_param',
                                    #shape=[1] + [self.K] + [1] * (len(input_shape) - 1),# + [1],
                                    #initializer=tf.keras.initializers.Ones,
                                    #regularizer=None,#tf.keras.regularizers.l2(self.lamb),
                                    #dtype = tf.float32,
                                    #trainable=True,
                                    #aggregation=tf.VariableAggregation.MEAN)
        
        self.k_param = self.add_weight(name='k_param',
                                    shape=[1] + [self.K] + [1] * (len(input_shape) - 2)+[input_shape[-1]//2] + [2],
                                    initializer=tf.keras.initializers.Ones,
                                    regularizer=None,#tf.keras.regularizers.l1(self.lamb),
                                    dtype = tf.float32,
                                    trainable=True,
                                    aggregation=tf.VariableAggregation.MEAN)
        
        init_shp = list(self.k_param.shape)
        init_shp[-1] = 1
        
        initialization = tf.concat([tf.random.uniform(init_shp,minval=0.001,maxval=10),
                                    tf.random.uniform(init_shp,minval=0.001,maxval=10)],axis=-1)
        self.k_param.assign(initialization)
        
        #self.omega_param = self.add_weight(name='omega_param',
        #                            shape=[1] + [self.K] + [1] * (len(input_shape) - 2)+[input_shape[-1]//2],
        #                            initializer=tf.keras.initializers.Zeros,
        #                            regularizer=None,#tf.keras.regularizers.l1(self.lamb),
        #                            dtype = tf.float32,
        #                            trainable=False,
        #                            aggregation=tf.VariableAggregation.MEAN)
        
        #self.omega_param.assign(tf.random.uniform(self.omega_param.shape,minval=0.001,maxval=2*np.pi))

        #self.mult_param = self.add_weight(name='mult_param',
        #                            shape=[1] + [self.K] + [1] * (len(input_shape) - 1),# + [1],
        #                            initializer=tf.keras.initializers.Ones,
        #                            regularizer=None,#tf.keras.regularizers.l1(self.lamb),
        #                            dtype = tf.float32,
        #                            trainable=False,
        #                            aggregation=tf.VariableAggregation.MEAN)

        #init_shp = self.a_param.shape
        #self.a_param.assign(tf.random.normal(init_shp,mean=1.0,stddev=0.2))
        #self.b_param.assign(tf.random.normal(init_shp,mean=1.0,stddev=0.2))
        
        #initialization_mult = tf.range(1,self.K+1,1,dtype=tf.float32)*0.1
        #initialization_mult = tf.reshape(initialization_mult,init_shp)
        
        #self.mult_param.assign(initialization_mult)

        #self.sign = self.add_weight(name='sign',
        #                                     shape=[1] * len(input_shape),
        #                                     initializer=tf.keras.initializers.GlorotNormal,
        #                                     regularizer=None,#tf.keras.regularizers.l2(self.lamb),
        #                                     trainable=True,
        #                                     dtype = tf.float32,
        #                                     aggregation=tf.VariableAggregation.MEAN)

        self.affine_params = self.add_weight(name='affine_params',
                                             shape=[1] * len(input_shape)+[2],
                                             initializer=tf.keras.initializers.Zeros,
                                             regularizer=None,#tf.keras.regularizers.l2(self.lamb),
                                             trainable=True,
                                             dtype = tf.float32,
                                             aggregation=tf.VariableAggregation.MEAN)

        init_shp =  [1] * (len(input_shape)+1)
        initialization = tf.concat([tf.zeros(init_shp) ,
        #                            tf.zeros(init_shp),
        #                            tf.zeros(init_shp),
                                    tf.zeros(init_shp)], axis=-1)
        #self.affine_params.assign(initialization)
        self.dropout = tf.keras.layers.Dropout(0.0)
        self.tile_shp = [1] + [self.K] + [1] * (len(input_shape) - 1)
        self.tile_shp_k = [1] + [self.K] + [1] * (len(input_shape))
        self.zero = self.add_weight(name='zero',shape=[1] * len(input_shape),trainable=False)
        self.zero.assign(tf.zeros([1] * len(input_shape), dtype=tf.float32))
        self.i = tf.complex(tf.constant(0.0),tf.constant(1.0))
        self.one = self.add_weight(name='one',shape=[1] * len(input_shape),trainable=False)
        self.one.assign(tf.ones([1] * len(input_shape), dtype=tf.float32))

    def apply_relus(self, inputs,training=False):
        
        #a_param = self.a_param
        #b_param = self.b_param
        k_param = self.k_param#tf.tile(self.k_param,self.tile_shp_k)
        #omega_param = self.omega_param
        
        #A_tmp = tf.math.sqrt(a_param**2+b_param**2)
        #norm = tf.reduce_sum(a_param,axis=1,keepdims=True)+1e-8
        #a_param = a_param/norm
        #norm = tf.reduce_sum(b_param,axis=1,keepdims=True)+1e-8
        #b_param = b_param/norm
        #a_param = a_param/(tf.sqrt(2.0)*norm)#tf.where(tf.abs(a_param)<1e-6,tf.ones_like(a_param)*1e-6,a_param)
        
        #if training:
        #    a_param = a_param*tf.random.normal(a_param.shape,mean=1.0,stddev=0.1)
        #    b_param = b_param*tf.random.normal(b_param.shape,mean=1.0,stddev=0.1)
        #    k_param = k_param*tf.random.normal(k_param.shape,mean=1.0,stddev=0.1)
        
        
        input_real = inputs[...,:inputs.shape[-1]//2]
        input_imag = inputs[...,inputs.shape[-1]//2:]
        
        #if training:
        #    t = tf.random.normal(omega_param.shape,mean=1.0,stddev=0.2)
        #else:
        #    t = tf.ones_like(omega_param)
        
        phase_real = input_real
        phase_imag = input_imag
        
        real = k_param[...,0]*phase_real-k_param[...,1]*phase_imag#-omega_param
        imag = k_param[...,0]*phase_imag+k_param[...,1]*phase_real
        real = tf.where(tf.math.is_finite(real),real,tf.ones_like(real)*1e-8)
        imag = tf.where(tf.math.is_finite(imag),imag,tf.ones_like(imag)*1e-8)


        phi = tf.where(tf.logical_and(real>0,imag>0),tf.math.atan2(imag,real),tf.zeros_like(imag))[0]
        x = tf.exp(tf.complex(tf.zeros_like(phi),phi))#,sign(self.sign)*phi))

        x_real = input_real*tf.math.real(x)
        x_imag = input_imag*tf.math.imag(x)

        x = tf.nn.sigmoid(self.affine_params[...,0])*tf.concat([x_real,x_imag],axis=-1)+(1-tf.nn.sigmoid(self.affine_params[...,0]))*tf.nn.relu(inputs)

        #x += inputs*self.affine_params[...,0]
        #x += self.affine_params[...,1]
        
        return x

    def call(self, inputs,training=False):
        x = self.apply_relus(inputs,training)

        return x

class CReLU(Layer):
    def __init__(self):
        super(CReLU, self).__init__()

    def build(self, input_shape):
        self.act = tf.keras.layers.Activation("relu")


    def call(self,inputs,training=False):

        input_real = inputs[...,:inputs.shape[-1]//2]
        input_imag = inputs[...,inputs.shape[-1] // 2:]

        out_real = self.act(input_real)
        out_imag = self.act(input_imag)

        return tf.concat([out_real,out_imag],axis=-1)
