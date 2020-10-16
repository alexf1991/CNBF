#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#

from tensorflow.keras import backend as K
import sys; sys.path.append('.')
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
from .init import ComplexInit, ComplexIndependentFilters
from .bn import ComplexBN as complex_normalization
from .bn import sqrt_init
import numpy as np
import tensorflow as tf


class ComplexDense(Layer):
    """Regular complex densely-connected NN layer.
    `Dense` implements the operation:
    `real_preact = dot(real_input, real_kernel) - dot(imag_input, imag_kernel)`
    `imag_preact = dot(real_input, imag_kernel) + dot(imag_input, real_kernel)`
    `output = activation(K.concatenate([real_preact, imag_preact]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    AN ERROR MESSAGE IS PRINTED.
    # Arguments
        units: Positive integer, dimensionality of each of the real part
            and the imaginary part. It is actualy the number of complex units.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
            By default it is 'complex'.
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
    # Input shape
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        For a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        data_format = K.image_data_format()
        kernel_shape = [input_dim, 2*self.units]

        fan_in = tf.cast(kernel_shape[-2],tf.float32)
        fan_out = tf.cast(kernel_shape[-1],tf.float32)
        #if self.init_criterion == 'he':
        #    s = tf.sqrt(1. / fan_in)
        #elif self.init_criterion == 'glorot':
        #    s = tf.sqrt(1. / (fan_in + fan_out))
        #rng = RandomStreams(seed=self.seed)

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * K.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * K.sin(phase)"""

        # Initialization using euclidean representation:
        #def init_w_real(shape, dtype=None):
            #return rng.normal(
                #size=kernel_shape,
                #avg=0,
                #std=s,
                #dtype=dtype
            #)
        #def init_w_imag(shape, dtype=None):
            #return rng.normal(
                #size=kernel_shape,
                #avg=0,
                #std=s,
                #dtype=dtype
            #)
        if self.kernel_initializer in {'complex', 'complex_independent'}:
            kls = {'complex':             ComplexInit,
                   'complex_independent': ComplexIndependentFilters}[self.kernel_initializer]
            kern_init = kls(
                kernel_size=[1,1],
                input_dim=input_dim,
                weight_dim=2,
                nb_filters=self.units,
                criterion=self.init_criterion,
                is_dense=True
            )
        else:
            kern_init = self.kernel_initializer

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=kern_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]
        real_kernel = self.kernel[:,:input_dim]
        imag_kernel = self.kernel[:,input_dim:]
        cat_kernels_4_real = K.concatenate(
            [real_kernel, -imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = K.concatenate(
            [imag_kernel, real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = K.dot(inputs, cat_kernels_4_complex)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

