# -*- coding:utf-8 -*-
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def activation(x, activation_fn='leaky') :
    assert activation_fn in ['relu', 'leaky', 'tanh', 'sigmoid', 'swish', None]
    if activation_fn == 'leaky':
        x = lrelu(x)

    if activation_fn == 'relu':
        x = relu(x)

    if activation_fn == 'sigmoid':
        x = sigmoid(x)

    if activation_fn == 'tanh' :
        x = tanh(x)

    if activation_fn == 'swish' :
        x = swish(x)

    return x

def lrelu(x, alpha=0.01) :
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def tanh(x) :
    return tf.tanh(x)

def swish(x) :
    return x * sigmoid(x)

def conv(h, filters, kernel_size=3, strides=2, pad=0, normal_weight_init=False, activation_fn='leaky'):

    h = tf.pad(h, [[0,0], [pad, pad], [pad, pad], [0,0]])

    if normal_weight_init:
        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                   kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    else:
        if activation_fn == 'relu' :
            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
        else:
            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)

    h = activation(h, activation_fn)
    
    return h

def deconv(h, filters, kernel_size=3, strides=2, normal_weight_init=False, activation_fn='leaky'):

    if normal_weight_init:
        h = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h) 
    else:
        if activation_fn == "relu":
            h = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                                kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
        else:
            h = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                                kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    h = activation(h, activation_fn)
    return h

def resblock(h_init, filters, kernel_size=3, strides=1, pad=1, dropout_ratio=0.0, normal_weight_init=False, norm_fn='instance'):

    h = tf.pad(h_init, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

    if normal_weight_init:
        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                   kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    else:
        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    if norm_fn == "instance":
        h = InstanceNormalization()(h)
    if norm_fn == "batch":
        h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

    if normal_weight_init:
        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                   kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    else:
        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h)
    if norm_fn == "instance":
        h = InstanceNormalization()(h)
    if norm_fn == "batch":
        h = tf.keras.layers.BatchNormalization()(h)

    if dropout_ratio > 0.0:
        h = tf.keras.layers.Dropout(dropout_ratio)(h)

    return h + h_init

def gaussian_noise_layer(input):
    gaussian_random_vector = tf.keras.layers.GaussianNoise(stddev=1.0)(input)
    return gaussian_random_vector