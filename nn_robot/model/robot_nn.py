import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nn_util import *
from util import *
import nvtx.plugins.tf as nvtx_tf
ENABLE_NVTX = True


@nvtx_tf.ops.trace(message='Projector Network', grad_message='Projector grad',
                   domain_name='Forward', grad_domain_name='Gradient',
                   enabled=ENABLE_NVTX, trainable=True)
def create_robot_new(x, is_training,arm):
    batch_size = [tf.shape(x)[0]]
    x, nvtx_context = nvtx_tf.ops.start(x, message='{}_Dense'.format(arm),
        grad_message='{}_Dense grad'.format(arm), domain_name='Forward',
        grad_domain_name='Gradient', trainable=True, enabled=ENABLE_NVTX)
    with tf.variable_scope("Dense"):
        x = tf.layers.dense(x,5*6*128)       
        x = batch_norm(x, is_training) 
        x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
        x = tf.reshape(x, (-1, 5,6,128))
        h_v = [9,18, 36, 72, 143, 285, 570] 
        w_v = [12,24, 48, 95, 190, 380, 760]
        filters_v = [128,128, 128, 64, 32, 16, 1]
        kernels_v = [3,3,3,5,5,5,5]
        n_layers = len(filters_v)
        x = nvtx_tf.ops.end(x, nvtx_context)

        for k in range(n_layers):
            print("------- Layer: {}".format(k))
            with tf.variable_scope("layer_{}".format(k)):
                stddev = np.sqrt(2 / (kernels_v[k] ** 2 * filters_v[k])) 
                print("\nstdev: {}".format(stddev))
                input_channels = x.get_shape().as_list()[-1]

                x, nvtx_context = nvtx_tf.ops.start(x, message="{}_Deconv_{}".format(arm,k), grad_message="{}_Deconv_{} grad".format(arm,k),
                    domain_name="Forward", grad_domain_name="Gradient", enabled=ENABLE_NVTX)

                w = get_weights([kernels_v[k], kernels_v[k], filters_v[k], input_channels], stddev, name="w_deconv")
                #b = get_biases([filters_v[k]], name="b_deconv")
                output_shape = tf.concat([batch_size, [h_v[k], w_v[k], filters_v[k]]], axis=0)
                 #x = tf.nn.bias_add(deconv2d(x, w, 2, output_shape),b)
                x = deconv2d(x, w, 2, output_shape)
                x = batch_norm(x, is_training, "bn_deconv")
                x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU_deconv")  
                
                x = nvtx_tf.ops.end(x, nvtx_context)


                x, nvtx_context = nvtx_tf.ops.start(x, message="{}_Conv1_{}".format(arm,k), grad_message="{}_Conv1_{} grad".format(arm,k),
                    domain_name="Forward", grad_domain_name="Gradient", enabled=ENABLE_NVTX)

                input_channels = x.get_shape().as_list()[-1]
                w = get_weights([kernels_v[k], kernels_v[k], input_channels, filters_v[k]], stddev, name="w_conv")
                #b = get_biases([filters_v[k]], name="b_conv")
                #x = tf.nn.bias_add(conv2d(x,w, stride = 1),b)
                x = conv2d(x,w, stride = 1)
                x = batch_norm(x, is_training, "bn_conv")
                x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")

                x = nvtx_tf.ops.end(x, nvtx_context)


                x, nvtx_context = nvtx_tf.ops.start(x, message="{}_Conv2_{}".format(arm,k), grad_message="{}_Conv2_{} grad".format(arm,k),
                    domain_name="Forward", grad_domain_name="Gradient", enabled=ENABLE_NVTX)

                input_channels = x.get_shape().as_list()[-1]
                w = get_weights([kernels_v[k], kernels_v[k], input_channels, filters_v[k]], stddev, name="w_conv2")
                #b = get_biases([filters_v[k]], name="b_conv")
                #x = tf.nn.bias_add(conv2d(x,w, stride = 1),b)
                x = conv2d(x,w, stride = 1)
                if k != (n_layers - 1):
                    x = batch_norm(x, is_training, "bn_conv2")
                    x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
                print("output shape: {}".format(x.shape))
                x = nvtx_tf.ops.end(x, nvtx_context)


    print("\n\nBuilt model, output: {}".format(x.get_shape()))
    return x

def create_robot_deeper(x, is_training):
    batch_size = [tf.shape(x)[0]]
    with tf.variable_scope("Dense"):
        x = tf.layers.dense(x,5*6*256)       
        x = batch_norm(x, is_training) 
        x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
        x = tf.reshape(x, (-1, 5,6,256))
        h_v = [9,18, 36, 72, 143, 285, 570] 
        w_v = [12,24, 48, 95, 190, 380, 760]
        filters_v = [256,128, 128, 64, 64, 32, 1]
        kernels_v = [3,3,3,5,5,5,5]
        n_layers = len(filters_v)
        for k in range(n_layers):
            print("------- Layer: {}".format(k))
            with tf.variable_scope("layer_{}".format(k)):
                if k == (n_layers-1):
                    stddev = np.sqrt(1 / (kernels_v[k] ** 2 * filters_v[k])) 
                else:
                    stddev = np.sqrt(2 / (kernels_v[k] ** 2 * filters_v[k])) 

                print("\nstdev: {}".format(stddev))
                input_channels = x.get_shape().as_list()[-1]

                w = get_weights([kernels_v[k], kernels_v[k], filters_v[k], input_channels], stddev, name="w_deconv")
                #b = get_biases([filters_v[k]], name="b_deconv")
                output_shape = tf.concat([batch_size, [h_v[k], w_v[k], filters_v[k]]], axis=0)
                 #x = tf.nn.bias_add(deconv2d(x, w, 2, output_shape),b)
                x = deconv2d(x, w, 2, output_shape)
                x = batch_norm(x, is_training, "bn_deconv")
                x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU_deconv")  
                
                input_channels = x.get_shape().as_list()[-1]
                w = get_weights([kernels_v[k], kernels_v[k], input_channels, filters_v[k]], stddev, name="w_conv")
                #b = get_biases([filters_v[k]], name="b_conv")
                #x = tf.nn.bias_add(conv2d(x,w, stride = 1),b)
                x = conv2d(x,w, stride = 1)
                if k != (n_layers - 1):
                    x = batch_norm(x, is_training, "bn_conv")
                    x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
                print("output shape: {}".format(x.shape))

    print("\n\nBuilt model, output: {}".format(x.get_shape()))
    return x

      
def create_robot(x, is_training):
    batch_size = [tf.shape(x)[0]]
    with tf.variable_scope("Dense"):
        x = tf.layers.dense(x,5*6*128)       
        x = batch_norm(x, is_training) 
        x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
        x = tf.reshape(x, (-1, 5,6,128))
        h_v = [9,18, 36, 72, 143, 285, 570] 
        w_v = [12,24, 48, 95, 190, 380, 760]
        filters_v = [128,128, 128, 64, 32, 16, 1]
        kernels_v = [3,3,3,5,5,5,5]
        n_layers = len(filters_v)
        for k in range(n_layers):
            print("------- Layer: {}".format(k))
            with tf.variable_scope("layer_{}".format(k)):
                stddev = np.sqrt(2 / (kernels_v[k] ** 2 * filters_v[k])) 
                print("\nstdev: {}".format(stddev))
                input_channels = x.get_shape().as_list()[-1]

                w = get_weights([kernels_v[k], kernels_v[k], filters_v[k], input_channels], stddev, name="w_deconv")
                #b = get_biases([filters_v[k]], name="b_deconv")
                output_shape = tf.concat([batch_size, [h_v[k], w_v[k], filters_v[k]]], axis=0)
                 #x = tf.nn.bias_add(deconv2d(x, w, 2, output_shape),b)
                x = deconv2d(x, w, 2, output_shape)
                x = batch_norm(x, is_training, "bn_deconv")
                x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU_deconv")  
                
                input_channels = x.get_shape().as_list()[-1]
                w = get_weights([kernels_v[k], kernels_v[k], input_channels, filters_v[k]], stddev, name="w_conv")
                #b = get_biases([filters_v[k]], name="b_conv")
                #x = tf.nn.bias_add(conv2d(x,w, stride = 1),b)
                x = conv2d(x,w, stride = 1)
                if k != (n_layers - 1):
                    x = batch_norm(x, is_training, "bn_conv")
                    x = tf.nn.leaky_relu(x, alpha = 0.2, name = "LeakyReLU")
                print("output shape: {}".format(x.shape))

    print("\n\nBuilt model, output: {}".format(x.get_shape()))
    return x

      

