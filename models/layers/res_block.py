import tensorflow as tf


class ResBlockBasicLayer(tf.keras.layers.Layer):
    def __init__(self,n_layers,n_channels,
                 stride=1,
                 kernel_regularizer = tf.keras.regularizers.l2(2e-4),
                 kernel_initializer = tf.keras.initializers.he_normal(),
                 dropout=0.0,
                 name=''):
        super(ResBlockBasicLayer, self).__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.stride = stride
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.dropout = dropout
        self.name_op = name
        
    def build(self,input_shape):

 
        self.layers = []


        self.shortcut = tf.keras.layers.Conv2D(self.n_channels,
                                    kernel_size = (1,1),
                                    strides = (self.stride, self.stride),
                                    padding = "same",
                                    use_bias = False,
                                    name = self.name_op+'_sc_conv',
                                    kernel_regularizer = self.kernel_regularizer,
                                    kernel_initializer = self.kernel_initializer)



        for i in range(self.n_layers):

            self.layers.append((tf.keras.layers.BatchNormalization(name=self.name_op+'_bn_'+str(i),axis=-1),1))
            self.layers.append((tf.keras.layers.Activation('relu'),0))
            if i == 0:
                self.layers.append((tf.keras.layers.Conv2D(self.n_channels,
                                        kernel_size=(3, 3),
                                        strides=(self.stride, self.stride),
                                        padding="same",
                                        use_bias=False,
                                        name=self.name_op+'_conv_' + str(i),
                                        kernel_regularizer = self.kernel_regularizer,
                                        kernel_initializer = self.kernel_initializer,
                                        activation=None),0))
            else:
                self.layers.append((tf.keras.layers.Conv2D(self.n_channels,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            use_bias=False,
                            name=self.name_op+'_conv_' + str(i),
                            kernel_regularizer = self.kernel_regularizer,
                            kernel_initializer = self.kernel_initializer,
                            activation=None),0))
            if self.dropout > 0.0:
                self.layers.append((tf.keras.layers.Dropout(rate=self.dropout),1))
            
            if (i+1) % 2 == 0:
                self.layers.append((tf.keras.layers.Add(),2))



    def call(self,input,training=False):
        x = input
        sc = self.shortcut(input)
        
        for layer in self.layers:
            if layer[1]==0:
                x = layer[0](x)
            elif layer[1]==1:
                x = layer[0](x,training)
            elif layer[1]==2:
                x = layer[0]([sc,x])
                sc = x

        return x
