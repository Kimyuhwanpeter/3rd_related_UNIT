# -*- coding:utf-8 -*-
from layer_ops import *
import tensorflow as tf
# https://github.com/taki0112/UNIT-Tensorflow/blob/master/UNIT.py
# ENCODERS
def encoder(input_shape=(256, 256, 3), filters=64):
    # input: 256 256
    channel = filters
    h = inputs = tf.keras.Input(input_shape)

    h = conv(h, channel, kernel_size=7, strides=1, pad=3, normal_weight_init=True, activation_fn="leaky")

    for i in range(1, 3):
        h = conv(h, channel*2, kernel_size=3, strides=2, pad=1, normal_weight_init=True, activation_fn="leaky")
        channel *=2

    # channel = 256
    for i in range(0, 3):
        h = resblock(h, channel, kernel_size=3, strides=1, pad=1, dropout_ratio=0.0,
                     normal_weight_init=True,norm_fn="instance")

    # [64, 64, 256]
    return tf.keras.Model(inputs=inputs, outputs=h)

# Shared residual-blocks
def share_encoder(input_shape=(64, 64, 256)):

    channel = 64 * pow(2, 3-1)
    h = inputs = tf.keras.Input(input_shape)
    for i in range(0, 1):
        h = resblock(h, channel, kernel_size=3, strides=1, pad=1, dropout_ratio=0.0, 
                     normal_weight_init=True, norm_fn="instance")
    h = gaussian_noise_layer(h)

    # [64, 64, 256]
    return tf.keras.Model(inputs=inputs, outputs=h)

def share_generator(input_shape=(64, 64, 256)):
    channel = 64 * pow(2, 3-1)
    h = inputs = tf.keras.Input(input_shape)
    for i in range(0, 1):
        h = resblock(h, channel, kernel_size=3, strides=1, pad=1, dropout_ratio=0.0, 
                     normal_weight_init=True, norm_fn="instance")
    return tf.keras.Model(inputs=inputs, outputs=h)

def generator(input_shape=(64, 64, 256)):

    channel = 64 * pow(2, 3-1)
    h = inputs = tf.keras.Input(input_shape)
    for i in range(0, 3):
        h = resblock(h, channel, kernel_size=3, strides=1, pad=1, dropout_ratio=0.0, 
                     normal_weight_init=True, norm_fn="instance")

    for i in range(0, 3-1):
        h = deconv(h, channel//2, kernel_size=3, strides=2, normal_weight_init=True, activation_fn="leaky")
        channel = channel // 2

    h = deconv(h, 3, kernel_size=1, strides=1, normal_weight_init=True, activation_fn="tanh")
    # [256, 256, 3]
    return tf.keras.Model(inputs=inputs, outputs=h)

# translation은 main 함수에 짜라

def discirminator(input_shape=(256, 256, 3)):
    channel = 64
    h = inputs = tf.keras.Input(input_shape)

    h = conv(h, channel, kernel_size=3, strides=2, pad=1, normal_weight_init=True, activation_fn="leaky")

    for i in range(1, 6):
        h = conv(h, channel*2, kernel_size=3, strides=2, pad=1, normal_weight_init=True, activation_fn="leaky")
        channel *= 2
    h = conv(h, 1, kernel_size=1, strides=1, pad=0, normal_weight_init=True, activation_fn="leaky")

    return tf.keras.Model(inputs=inputs, outputs=h)

def generator_loss(fake, smoothing=False, use_lsgan=False) :
    if use_lsgan :
        if smoothing :
            loss = tf.reduce_mean(tf.math.squared_difference(fake, 0.9)) * 0.5
        else :
            loss = tf.reduce_mean(tf.math.squared_difference(fake, 1.0)) * 0.5
    else :
        if smoothing :
            fake_labels = tf.fill(tf.shape(fake), 0.9)
        else :
            fake_labels = tf.ones_like(fake)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    return loss

def discriminator_loss(real, fake, smoothing=False, use_lasgan=False) :
    if use_lasgan :
        if smoothing :
            real_loss = tf.reduce_mean(tf.math.squared_difference(real, 0.9)) * 0.5
        else :
            real_loss = tf.reduce_mean(tf.math.squared_difference(real, 1.0)) * 0.5

        fake_loss = tf.reduce_mean(tf.square(fake)) * 0.5
    else :
        if smoothing :
            real_labels = tf.fill(tf.shape(real), 0.9)
        else :
            real_labels = tf.ones_like(real)

        fake_labels = tf.zeros_like(fake)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    loss = real_loss + fake_loss

    return loss

def KL_divergence(mu) :
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)

    return loss

def L1_loss(x, y) :
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss