import tensorflow as tf
from tensorflow.python.client import device_lib 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import os
import time
import pandas as pd
import pickle as pkl 
from scipy.stats import skewnorm
import sys
sys.path.append("/b/home/icube/lsestini/sestini/project_module")
from project_main import *

my_folder_rooth = "/b/home/icube/lsestini/sestini/vtk_robot/"
 
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if "GPU" in str(x.device_type)])
 
def get_weights(shape):
    #weights shape = [input channels, output channels, w_h, w_w]
    return tf.get_variable(name="W", shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.02))

def get_biases(shape):
    #bias shape = [output channels]
    return tf.get_variable(name="b", shape=shape, initializer=tf.zeros_initializer())

def conv2d(input, kernel, filters, stride, is_training, padding = "SAME", name = "Conv"):
    with tf.variable_scope(name):
        batch_size, w, h, input_channels = input.get_shape().as_list()
        W = get_weights([kernel,kernel, input_channels, filters])
        b = get_biases([filters])
        conv = tf.nn.conv2d(input, filter = W, strides = [1,stride,stride,1], padding = padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), (-1,conv.get_shape().as_list()[1],conv.get_shape().as_list()[2],conv.get_shape().as_list()[3]))
        bn_conv = batch_norm(conv, is_training)
        x = tf.nn.leaky_relu(bn_conv, alpha = 0.2, name = "LeakyReLU")
    return x


def conv2d_transp_bn_biases(input_x, kernel, output_shape, filters, stride, is_training, padding = "SAME", name = "Deconv"):
    with tf.variable_scope(name):
        batch_size, w, h, input_channels = input_x.get_shape().as_list()
        W = get_weights([kernel,kernel,filters,input_channels])
        b = get_biases(filters)
        print(W.get_shape())
        conv = tf.nn.conv2d_transpose(input_x, filter=W, output_shape=output_shape, strides=[1,stride,stride,1], padding=padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), (-1,conv.get_shape().as_list()[1],conv.get_shape().as_list()[2],conv.get_shape().as_list()[3]))
        bn_conv = batch_norm(conv, is_training)
        x = tf.nn.leaky_relu(bn_conv, alpha = 0.2, name = "LeakyReLU")
    return x


def batch_norm(x, is_training):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, scope='bn')

def model1_bn_biases(input, is_training):
    with tf.name_scope('Model'): 
        print('Building model configuration')
        with tf.variable_scope("Dense"):
            x = tf.compat.v1.layers.dense(input,5*6*128, activation = tf.nn.relu)
        with tf.variable_scope("Reshape"):
            x = tf.reshape(x, (-1,5,6,128))
            x = batch_norm(x, is_training)
        with tf.variable_scope("Deconvs"):
            h_v = [9,18, 36, 72, 143, 285, 570] 
            w_v = [12,24, 48, 95, 190, 380, 760]
            filters_v = [128,64, 32, 16, 8, 4, 1]
            kernels_v = [3,3,3,5,5,5,5]
            for k in range(len(h_v)):
                curr_batch = tf.shape(x)[0:1]
                output_shape = tf.concat([curr_batch,[h_v[k], w_v[k], filters_v[k]]], axis = 0)
                x = conv2d_transp_bn_biases(x, output_shape=output_shape, kernel=kernels_v[k], filters=filters_v[k], stride=2, padding='SAME', is_training = is_training, name='deconv_{}_1'.format(k))   
                print("Building model, shape: {}\n\n\n".format(x.get_shape()))
            with tf.variable_scope("OutputMask"):
                logits = x
                out = tf.math.sigmoid(x)
        print("\n\nBuilt model, output: {}".format(out.get_shape()))
        return logits, out

def model2_bn_biases(input, is_training):
    with tf.name_scope('Model'): 
        print('Building model configuration')
        with tf.variable_scope("Dense"):
            x = tf.compat.v1.layers.dense(input,5*6*128, activation = tf.nn.relu)
        with tf.variable_scope("Reshape"):
            x = tf.reshape(x, (-1,5,6,128))
            x = batch_norm(x, is_training)
        with tf.variable_scope("Deconvs"):
            h_v = [9,18, 36, 72, 143, 285, 570] 
            w_v = [12,24, 48, 95, 190, 380, 760]
            filters_v = [128,64, 32, 16, 8, 4, 1]
            kernels_v = [3,3,3,5,5,5,5]
            for k in range(len(h_v)):
                curr_batch = tf.shape(x)[0:1]
                output_shape = tf.concat([curr_batch,[h_v[k], w_v[k], filters_v[k]]], axis = 0)
                x = conv2d_transp_bn_biases(x, output_shape=output_shape, kernel=kernels_v[k], filters=filters_v[k], stride=2, padding='SAME', is_training = is_training, name='deconv_{}_1'.format(k))   
                x = conv2d(x, kernel=kernels_v[k], filters=filters_v[k], stride=1, is_training = is_training, padding='SAME', name='conv_{}_2'.format(k))
                print("Building model, shape: {}\n\n\n".format(x.get_shape()))
            with tf.variable_scope("OutputMask"):
                logits = x
                out = tf.math.sigmoid(x)
        print("\n\nBuilt model, output: {}".format(out.get_shape()))
        return logits, out

def loss_function(logits, y):
    print(logits.get_shape(), y.get_shape())
    print(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y, name = "BinaryCrossEntropy").get_shape())
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels =  y, name = "BinaryCrossEntropy"))
    print(loss.get_shape())
    return loss

def loss_function2(logits, y,out):
    print(logits.get_shape(), y.get_shape())
    print(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y, name = "BinaryCrossEntropy").get_shape())
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels =  y, name = "BinaryCrossEntropy")) + tf.reduce_sum(tf.ones_like(labels) - tf.round(out))/(tf.cast(tf.size(labels), tf.float32))
    print(loss.get_shape())
    return loss

def f1_score(out, GT, smooth = 1):
    out_mask = tf.round(out)
    inters = tf.reduce_sum(out_mask*GT, axis = (1,2,3))
    union = tf.reduce_sum(out_mask, axis = (1,2,3)) + tf.reduce_sum(GT, axis = (1,2,3))
    f1 = tf.reduce_mean((2*inters + 1)/(union+smooth), axis = 0)
    print(f1.get_shape())
    return f1

def test_net():
    input_kine = gen_random((16,3))
    out_mask,accuracy_out = sess.run([out,accuracy], feed_dict={inputs:input_kine,labels:np.ones([input_kine.shape[0],570,760,1]),"phase:0":0})
    #     out_mask,accuracy_out = sess.run([out,accuracy], feed_dict={inputs:input_kine,labels:np.ones([input_kine.shape[0],513,705,1])})
    return out_mask, accuracy_out

def normalize(data,mm):
    max_ = mm[0]
    min_ = mm[1]
    return (data - min_)/(max_ - min_)

def normalize_kine(kine,arm,mm):
    mm = mm[arm]
    transl = normalize(kine[:,0],mm[0])
    rot = normalize(kine[:,1],mm[1])
    bend = normalize(kine[:,2],mm[2])
    #print("Max min for joint: ",[max(transl), min(transl)],[max(rot), min(rot)],[max(bend), min(bend)])
    kine = np.array([transl, rot, bend]).transpose()
    return kine 


def load_dataset_full(arm):
    folder_read =  "/b/home/icube/lsestini/sestini/proj_generator/dataset/training_dataset_sample/"
    kine_v = []
    proj_v = []
    i = 0
    for files in np.sort(os.listdir(folder_read)):
        dataset_sample_pd = pd.read_pickle(folder_read + files)
        n_i = len(dataset_sample_pd["kine"])
        kine_v_i = np.array(dataset_sample_pd["kine"])
        proj_v_i = np.array(dataset_sample_pd["proj"])
        print(kine_v_i.shape, proj_v_i.shape)
        kine_v = np.hstack((kine_v, kine_v_i))
        proj_v = np.hstack((proj_v, proj_v_i))
        i += 1
        print(i, files)
    n = len(kine_v)
    kine_v =  np.array([kine_v[i][arm] for i in range(n)])
    kine_v = normalize_kine(kine_v, arm)
    proj_v = np.array([proj_v[i][arm] for i in range(n)])
    proj_v = proj_v.reshape((proj_v.shape[0],proj_v.shape[1], proj_v.shape[2],1))
    shapek, shapei = (kine_v.shape, proj_v.shape)
    print("Loaded:\nshape Kine - {}\nshape Images - {}".format(shapek, shapei))
    return kine_v, proj_v

def train_test_split(kine,proj, test_size, valid_size): 
    X_train, X_test, y_train, y_test = sk.train_test_split(kine,proj,test_size=test_size, random_state = 42)
    n_valid = int(round(len(X_train)*valid_size))
    return X_train[:-n_valid], y_train[:-n_valid], X_train[-n_valid:], y_train[-n_valid:], X_test, y_test

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def gen_random(shape,):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i,j] = np.random.rand()
    return out

def count_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print("Total parameters: {}".format(total_parameters))

def skewnorm_sample(distr, x):
    prob = np.random.rand()
    index = (np.abs(np.asarray(distr) - prob)).argmin()
    return x[index]

def get_distributions(my_folder_rooth):    
    with open(my_folder_rooth + "project_utilities/joints_distr.pkl","rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        joints_distr = u.load()

    with open(my_folder_rooth + "project_utilities/max_min_joints.pkl","rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        mm = u.load()

    n_distr_samples = 5000

    x_tl = np.linspace(joints_distr[0][0][1], joints_distr[0][0][2], n_distr_samples)
    tl_distr = skewnorm.cdf(x_tl, *joints_distr[0][0][0])

    x_rl = np.linspace(joints_distr[1][0][1], joints_distr[1][0][2], n_distr_samples)
    rl_distr = skewnorm.cdf(x_rl, *joints_distr[1][0][0])

    x_bl = np.linspace(joints_distr[2][0][1], joints_distr[2][0][2], n_distr_samples)
    bl_distr = skewnorm.cdf(x_bl, *joints_distr[2][0][0])

    x_tr = np.linspace(joints_distr[3][0][1], joints_distr[3][0][2], n_distr_samples)
    tr_distr = skewnorm.cdf(x_tr, *joints_distr[3][0][0])

    x_rr1 = np.linspace(joints_distr[4][0][1], joints_distr[4][0][2], n_distr_samples)
    x_rr2 = np.linspace(joints_distr[4][1][1], joints_distr[4][1][2], n_distr_samples)
    x_rr3 = np.linspace(joints_distr[4][2][1], joints_distr[4][2][2], n_distr_samples)
    rr_distr1 = skewnorm.cdf(x_rr1, *joints_distr[4][0][0])
    rr_distr2 = skewnorm.cdf(x_rr2, *joints_distr[4][1][0])
    rr_distr3 = skewnorm.cdf(x_rr3, *joints_distr[4][2][0])


    x_br1 = np.linspace(joints_distr[5][0][1], joints_distr[5][0][2], n_distr_samples)
    x_br2 = np.linspace(joints_distr[5][1][1], joints_distr[5][1][2], n_distr_samples)
    br_distr1 = skewnorm.cdf(x_br1, *joints_distr[5][0][0])   
    br_distr2 = skewnorm.cdf(x_br2, *joints_distr[5][1][0])   

    distributions = [[tl_distr,x_tl], [rl_distr,x_rl], [bl_distr,x_bl], 
            [tr_distr,x_tr], [[rr_distr1,x_rr1],[rr_distr2,x_rr2],[rr_distr3,x_rr3]], [[br_distr1,x_br1],[br_distr2,x_br2]]]

    return distributions, mm

def draw_sample(distributions):
    tl = skewnorm_sample(distributions[0][0], distributions[0][1])
    rl = skewnorm_sample(distributions[1][0], distributions[1][1])
    bl = skewnorm_sample(distributions[2][0], distributions[2][1])
    tr = skewnorm_sample(distributions[3][0], distributions[3][1])
    rr_dist = np.random.choice([0,1,2])
    rr = skewnorm_sample(distributions[4][rr_dist][0], distributions[4][rr_dist][1])
    br_dist = np.random.choice([0,1])
    br = skewnorm_sample(distributions[5][br_dist][0], distributions[5][br_dist][1])
    return [[tl,rl,bl],[tr,rr,br]]

def generate_samples(my_folder_rooth, distributions, mm, n, arm):
    n_samples = n
    img_default = my_folder_rooth + "frame_ref.jpg"

    kine_v = []
    im_v = []
    i = 0
    print("Building dataset: {} samples".format(n_samples))
    time_s = time.time()
    while i in range(n_samples):
        kine = draw_sample(distributions)
        try:
            img_project_l, img_project_r, img_scope = project_dataset(img_default, kine, False)
            #img_project = combine_left_right(img_project_l, img_project_r)
            kine_v.append(kine[arm])
            im_v.append([img_project_l, img_project_r][arm])
            i += 1
        except:
            pass
        
    #print("elapsed time: {}s".format(round(time.time() - time_s)))
    kine_v =  np.array(kine_v)
    kine_v = normalize_kine(kine_v, arm, mm)
    proj_v = np.array(im_v)
    proj_v = proj_v.reshape((proj_v.shape[0],proj_v.shape[1], proj_v.shape[2],1))
    return kine_v, proj_v



dtype = tf.float32
learning_rate = 0.001  # The optimization initial learning rate

inputs = tf.placeholder(dtype = dtype, shape = (None,3), name = "RobotConfig")
labels = tf.placeholder(dtype = dtype, shape = (None, 570, 760, 1), name = "GTmask")
is_training = tf.placeholder(dtype = tf.bool, name = "phase")

logits, out = model2_bn_biases(inputs, is_training)
count_variables()

# Define the loss function, optimizer, and accuracy
loss = loss_function(logits, labels)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
training = optimizer.minimize(loss)
accuracy = f1_score(out, labels)

writer_tl = tf.summary.FileWriter("/b/home/icube/lsestini/sestini/logs/train_loss", tf.get_default_graph())
writer_ta = tf.summary.FileWriter("/b/home/icube/lsestini/sestini/logs/train_acc", tf.get_default_graph())
writer_va = tf.summary.FileWriter("/b/home/icube/lsestini/sestini/logs/valid_acc", tf.get_default_graph())
training_loss_summary = tf.summary.scalar("training_loss", loss)
training_accuracy_summary = tf.summary.scalar("training_accuracy", accuracy)
validation_accuracy_summary = tf.summary.scalar("validation_accuracy", accuracy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out_test, acc_test = test_net()
im = np.round(out_test[0].reshape(out_test[0].shape[0], out_test[0].shape[1]))
#plt.imshow(im, cmap=plt.cm.gray)
print("f1-score = {}".format(acc_test))




update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #control_dependencies add dependencies in brackets to ops created inside
    # Ensures that we execute the update_ops before performing the train_step
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
    training = optimizer.minimize(loss)

saver = tf.train.Saver(max_to_keep = 5)   
checkpoints_path = "/b/home/icube/lsestini/sestini/ckpts/"
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
epochs = 10000       # Total number of training epochs
batch_size = 32        # Training batch size
display_freq = 50      # Frequency of displaying the training results
display_freq_valid = 2
save_ckp_freq = 500
n_samples_train = 100*batch_size
n_samples_validation = 100
sess.run(init)
global_step = 0

# Number of training iterations in each epoch
num_tr_iter = int(n_samples_train / batch_size)
distributions,mm = get_distributions(my_folder_rooth)
arm = 1
print("Epochs: {}, Iterations per epoch: {}".format(epochs, num_tr_iter))
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = generate_samples(my_folder_rooth, distributions, mm, n_samples_train, arm)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = min((iteration + 1) * batch_size, len(y_train))
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        
        # Run optimization op (backprop)
        feed_dict_batch = {inputs: x_batch, labels: y_batch, "phase:0":1}
        #loss_out, _, logits_out, labels_out, out_out = sess.run([loss,training, logits, labels, out], feed_dict=feed_dict_batch)
        sess.run(training, feed_dict=feed_dict_batch)
        #print(loss_out)

        if global_step % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch, t_loss_summ, t_acc_summ  = sess.run([loss, accuracy, training_loss_summary, training_accuracy_summary],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
            writer_tl.add_summary(t_loss_summ, global_step) 
            writer_ta.add_summary(t_acc_summ, global_step)

        if global_step % save_ckp_freq == 0:
            saver.save(sess, checkpoints_path + "model-{}.ckpt".format(global_step))



    # Run validation after every epoch
    if global_step % display_freq_valid == 0:
        x_valid, y_valid = generate_samples(my_folder_rooth, distributions, mm, n_samples_validation, arm)
        feed_dict_valid = {inputs: x_valid, labels: y_valid,"phase:0":0}
        loss_valid, acc_valid, v_acc_summ = sess.run([loss, accuracy, validation_accuracy_summary], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
            format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')
        writer_va.add_summary(v_acc_summ, global_step)
