from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_weights(shape, std = 0.05, name = "W"):
    return tf.get_variable(name=name, shape=shape,initializer=tf.truncated_normal_initializer(stddev=std))

def get_weights_deconv(shape, std = 0.05, name = "W_deconv"):
    return tf.get_variable(name=name, shape=shape,initializer=tf.truncated_normal_initializer(stddev=std))

def get_biases(shape, name = "b"):
    #bias shape = [output channels]
    return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer())

def conv2d(x, W, stride):
    if type(stride) is int:
        stride = [stride]*2
    with tf.variable_scope("conv2d"):
        return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME', name="conv2d")

def conv_layer(x, scope, output_ch, layers):
    with tf.variable_scope(scope):
        input_ch = x.get_shape().as_list()[-1]
        w = get_weights([3, 3, input_ch, output_ch], 0.02, name="kernel")
        b = get_biases([output_ch], "bias")
        x = tf.nn.bias_add(conv2d(x, w, stride=1), b)
        x = tf.nn.relu(x, name="ReLU")
        layers[scope] = x
        return x, layers

def deconv2d(x, W,stride, output_shape):
    if type(stride) is int:
        stride = [stride]*2
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride[0], stride[1], 1], padding='SAME', name="conv2d_transpose")

def max_pool(x,k,s):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID')

def batch_norm(x, is_training, name = "bn"):
    return tf.layers.batch_normalization(x, center=True, scale=True, training=is_training, name=name)

def instance_norm(x, scope = "inst_norm"):
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5
        alpha = tf.get_variable("alpha", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))
        return x_ins * alpha + beta


@tf.custom_gradient
def my_round(x):

    # The custom gradient
    def grad(dy):
        return 1*dy

    # Return the result AND the gradient
    return tf.round(x), grad

def my_round2(x):
    return tf.where(x < 0.5, x*0.1, (x-0.5)*0.1 + 0.95)

