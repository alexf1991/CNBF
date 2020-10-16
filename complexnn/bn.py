#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi, Olexa Bilaniuk
#
# Note: The implementation of complex Batchnorm is based on
#       the Keras implementation of batch Normalization
#       available here:
#       https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py

import numpy as np
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
import tensorflow as tf

def sqrt_init(shape, dtype=None):
    value = (1 / tf.sqrt(2.0)) * tf.ones(shape)
    return value


def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)
def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)



def complex_standardization(input_centred, Vrr, Vii, Vri,
                            layernorm=False, axis=-1):
    
    ndim = K.ndim(input_centred)
    input_dim = tf.shape(input_centred)[axis] // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    if layernorm:
        variances_broadcast[0] = tf.shape(input_centred)[0]

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
    tau = Vrr + Vii
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s = tf.sqrt(delta) # Determinant of square root matrix
    t = tf.sqrt(tau + 2 * s)
    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/delta)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = tf.reshape(Wrr, variances_broadcast)
    broadcast_Wri = tf.reshape(Wri, variances_broadcast)
    broadcast_Wii = tf.reshape(Wii, variances_broadcast)

    cat_W_4_real = tf.concat([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = tf.concat([broadcast_Wri, broadcast_Wri], axis=axis)

    if (axis == 1 and ndim != 3) or ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif axis == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif axis == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = tf.concat([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output


def ComplexBN(input_centred, Vrr, Vii, Vri, beta,
               gamma_rr, gamma_ri, gamma_ii, scale=True,
               center=True, layernorm=False, axis=-1):

    ndim = K.ndim(input_centred)
    input_dim = tf.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            layernorm,
            axis=axis
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag
        
        broadcast_gamma_rr = tf.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = tf.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = tf.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = tf.concat([broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = tf.concat([broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if (axis == 1 and ndim != 3) or ndim == 2:
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif axis == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif axis == -1 and ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = tf.concat([centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = tf.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = tf.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


class ComplexBatchNormalization(Layer):
    """Complex version of the real domain 
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous complex layer at each batch,
    i.e. applies a transformation that maintains the mean of a complex unit
    close to the null vector, the 2 by 2 covariance matrix of a complex unit close to identity
    and the 2 by 2 relation matrix, also called pseudo-covariance, close to the 
    null matrix.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=2` in `ComplexBatchNormalization`.
        momentum: Momentum for the moving statistics related to the real and
            imaginary parts.
        epsilon: Small float added to each of the variances related to the
            real and imaginary parts in order to avoid dividing by zero.
        center: If True, add offset of `beta` to complex normalized tensor.
            If False, `beta` is ignored.
            (beta is formed by real_beta and imag_beta)
        scale: If True, multiply by the `gamma` matrix.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the real_beta and the imag_beta weight.
        gamma_diag_initializer: Initializer for the diagonal elements of the gamma matrix.
            which are the variances of the real part and the imaginary part.
        gamma_off_initializer: Initializer for the off-diagonal elements of the gamma matrix.
        moving_mean_initializer: Initializer for the moving means.
        moving_variance_initializer: Initializer for the moving variances.
        moving_covariance_initializer: Initializer for the moving covariance of
            the real and imaginary parts.
        beta_regularizer: Optional regularizer for the beta weights.
        gamma_regularizer: Optional regularizer for the gamma weights.
        beta_constraint: Optional constraint for the beta weights.
        gamma_constraint: Optional constraint for the gamma weights.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init',
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer              = sanitizedInitGet(beta_initializer)
        self.gamma_diag_initializer        = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer         = sanitizedInitGet(gamma_off_initializer)
        self.moving_mean_initializer       = sanitizedInitGet(moving_mean_initializer)
        self.moving_variance_initializer   = sanitizedInitGet(moving_variance_initializer)
        self.moving_covariance_initializer = sanitizedInitGet(moving_covariance_initializer)
        self.beta_regularizer              = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer        = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer         = regularizers.get(gamma_off_regularizer)
        self.beta_constraint               = constraints .get(beta_constraint)
        self.gamma_diag_constraint         = constraints .get(gamma_diag_constraint)
        self.gamma_off_constraint          = constraints .get(gamma_off_constraint)

    def build(self, input_shape):

        ndim = len(input_shape)

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        param_shape = (input_shape[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            regularizer=self.gamma_off_regularizer,
                                            constraint=self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vrr',
                                              trainable=False)
            self.moving_Vii = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vii',
                                              trainable=False)
            self.moving_Vri = self.add_weight(shape=param_shape,
                                              initializer=self.moving_covariance_initializer,
                                              name='moving_Vri',
                                              trainable=False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape[self.axis],),
                                               initializer=self.moving_mean_initializer,
                                               name='moving_mean',
                                               trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 2
        mu = tf.reduce_mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = tf.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = tf.reduce_mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon
            Vii = tf.reduce_mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = tf.reduce_mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        input_bn = ComplexBN(
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            axis=self.axis
        )
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean, mu, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centred = inputs - tf.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = inputs
                return ComplexBN(
                    inference_centred, self.moving_Vrr, self.moving_Vii,
                    self.moving_Vri, self.beta, self.gamma_rr, self.gamma_ri,
                    self.gamma_ii, self.scale, self.center, axis=self.axis
                )

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(input_bn,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':              sanitizedInitSer(self.beta_initializer),
            'gamma_diag_initializer':        sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer':         sanitizedInitSer(self.gamma_off_initializer),
            'moving_mean_initializer':       sanitizedInitSer(self.moving_mean_initializer),
            'moving_variance_initializer':   sanitizedInitSer(self.moving_variance_initializer),
            'moving_covariance_initializer': sanitizedInitSer(self.moving_covariance_initializer),
            'beta_regularizer':              regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer':        regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer':         regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint':               constraints .serialize(self.beta_constraint),
            'gamma_diag_constraint':         constraints .serialize(self.gamma_diag_constraint),
            'gamma_off_constraint':          constraints .serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ComplexFilterResponseNormalization(tf.keras.layers.Layer):
    """Filter response normalization layer.
    Filter Response Normalization (FRN), a normalization
    method that enables models trained with per-channel
    normalization to achieve high accuracy. It performs better than
    all other normalization techniques for small batches and is par
    with Batch Normalization for bigger batch sizes.
    Arguments
        axis: List of axes that should be normalized. This should represent the
              spatial dimensions.
        epsilon: Small positive float value added to variance to avoid dividing by zero.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        learned_epsilon: (bool) Whether to add another learnable
        epsilon parameter or not.
        name: Optional name for the layer
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model. This layer, as of now,
        works on a 4-D tensor where the tensor should have the shape [N X H X W X C]
        TODO: Add support for NCHW data format and FC layers.
    Output shape
        Same shape as input.
    References
        - [Filter Response Normalization Layer: Eliminating Batch Dependence
        in the training of Deep Neural Networks]
        (https://arxiv.org/abs/1911.09737)
    """

    def __init__(
        self,
        epsilon = 1e-6,
        axis = [1, 2],
        momentum = None,
        beta_initializer= "zeros",
        gamma_initializer = "ones",
        beta_regularizer = None,
        gamma_regularizer = None,
        beta_constraint = None,
        gamma_constraint = None,
        learned_epsilon = False,
        learned_epsilon_constraint = None,
        name = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = tf.math.abs(tf.cast(epsilon, dtype=self.dtype))
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.use_eps_learned = learned_epsilon
        self.supports_masking = True

        if self.use_eps_learned:
            self.eps_learned_initializer = tf.keras.initializers.Constant(1e-4)
            self.eps_learned_constraint = tf.keras.constraints.get(
                learned_epsilon_constraint
            )
            self.eps_learned = self.add_weight(
                shape=(1,),
                name="learned_epsilon",
                dtype=self.dtype,
                initializer=tf.keras.initializers.get(self.eps_learned_initializer),
                regularizer=None,
                constraint=self.eps_learned_constraint,
            )
        else:
            self.eps_learned_initializer = None
            self.eps_learned_constraint = None
        axis = [1,2]
        self._check_axis(axis)

    def build(self, input_shape):
        if len(tf.TensorShape(input_shape)) != 4:
            raise ValueError(
                """Only 4-D tensors (CNNs) are supported
        as of now."""
            )
        self._check_if_input_shape_is_none(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        epsilon = self.epsilon
        if self.use_eps_learned:
            epsilon += tf.math.abs(self.eps_learned)
            
        real = inputs[...,:inputs.shape[-1]//2]
        imag = inputs[...,inputs.shape[-1]//2:]
        nu2_real = tf.reduce_mean(real**2-imag**2, axis=self.axis, keepdims=True)
        nu2_real = tf.where(nu2_real != 0.0,nu2_real,tf.ones_like(nu2_real)*epsilon)
        nu2_imag = tf.reduce_mean(real*imag, axis=self.axis, keepdims=True)
        nu2_real = tf.where(nu2_imag != 0.0,nu2_imag,tf.ones_like(nu2_imag)*epsilon)
        normalized_inputs_real = real * tf.math.rsqrt(nu2_real + epsilon)
        normalized_inputs_imag = imag * tf.math.rsqrt(nu2_imag + epsilon)
        normalized_inputs = tf.concat([normalized_inputs_real,normalized_inputs_imag],axis=-1)
        return self.gamma * normalized_inputs + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "learned_epsilon": self.use_eps_learned,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            "learned_epsilon_constraint": tf.keras.constraints.serialize(
                self.eps_learned_constraint
            ),
        }
        base_config = super().get_config()
        return dict(**base_config, **config)

    def _create_input_spec(self, input_shape):
        ndims = len(tf.TensorShape(input_shape))
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError("Invalid axis: %d" % x)

        if len(self.axis) != len(set(self.axis)):
            raise ValueError("Duplicate axis: %s" % self.axis)

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        self.input_spec = tf.keras.layers.InputSpec(ndim=ndims, axes=axis_to_dim)

    def _check_axis(self, axis):
        if not isinstance(axis, list):
            raise TypeError(
                """Expected a list of values but got {}.""".format(type(axis))
            )
        else:
            self.axis = axis

        if self.axis != [1, 2]:
            raise ValueError(
                """FilterResponseNormalization operates on per-channel basis.
                Axis values should be a list of spatial dimensions."""
            )

    def _check_if_input_shape_is_none(self, input_shape):
        dim1, dim2 = input_shape[self.axis[0]], input_shape[self.axis[1]]
        if dim1 is None or dim2 is None:
            raise ValueError(
                """Axis {} of input tensor should have a defined dimension but
                the layer received an input with shape {}.""".format(
                    self.axis, input_shape
                )
            )

    def _add_gamma_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[-1]
        shape = [1, 1, 1, dim]
        # Initialize gamma with shape (1, 1, 1, C)
        self.gamma = self.add_weight(
            shape=shape,
            name="gamma",
            dtype=self.dtype,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
        )

    def _add_beta_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[-1]
        shape = [1, 1, 1, dim]
        # Initialize beta with shape (1, 1, 1, C)
        self.beta = self.add_weight(
            shape=shape,
            name="beta",
            dtype=self.dtype,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
        )
