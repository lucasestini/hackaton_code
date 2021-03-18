import os
import numpy as np
import time
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from nn_util import * 
from util import *
from robot_nn import *
from PIL import Image
import nvtx.plugins.tf as nvtx_tf
from nvtx.plugins.tf.estimator import NVTXHook



    
class model(object):
    def create_model(self, mask_l, mask_r, kine_l, kine_r, is_training):

        self.mask = tf.round(tf.clip_by_value(mask_l, 0., 1.))

        with tf.variable_scope("robot_l"):
            self.proj_logits_l = create_robot_new(kine_l, is_training,"Left")

        self.proj_sigm_l = tf.nn.sigmoid(self.proj_logits_l)

        self.pred_mask = tf.round(tf.clip_by_value(tf.round(self.proj_sigm_l),0., 1.))


        with tf.name_scope("cost"):
            self.cost_l = self._get_dice_coef_loss(self.proj_sigm_l, mask_l)


        with tf.name_scope("accuracy"):
            self.f1 = self._get_f1_accuracy(self.pred_mask, self.mask)


    def _get_f1_accuracy(self,y_pred, y_true):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
        return (numerator + 1e-6)/(denominator + 1e-6)

    def _dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.layers.flatten(y_true)
        y_pred_f = tf.layers.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


    def _get_dice_coef_loss(self, y_pred, y_true):
        return 1.-(self._dice_coef(y_true, y_pred))**2

    
    def _get_cost_ce(self,pred_log, y_true):
        beta = 0.55
        scale = 2. # to keep higher loss value
        beta_mult = beta/(1-beta)
        #loss = tf.nn.weighted_cross_entropy_with_logits(logits = pred_log, labels =  y_true, name = "BinaryCrossEntropy", pos_weight = beta_mult)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_log, labels =  y_true, name = "BinaryCrossEntropy")
        #return tf.reduce_mean(loss*(1-beta)*scale)
        return tf.reduce_mean(loss)

    def save(self, saver, sess, model_path, global_step):
        saver.save(sess, model_path + "model-{}.ckpt".format(global_step))
        print("Session saved at folder: {}".format(model_path))

    def restore(self, model_path, sess = None, saver = None, mode = "train"):
        return_ = False
        if sess is None:
            return_ = True
            sess = tf.Session()
        if mode == "test":
            vars_rest = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            vars_rest = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not "learning_rate" in var.name]
        vars_rest = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) # restore also lrates
        if saver is None:
            saver = tf.train.Saver(vars_rest)
        saver.restore(sess, model_path)
        print("Model robot restored from file: %s" % model_path)
        if return_:
            return sess  

    def restore_inference(self, ckpt, sess):
        vars_rest_left = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="robot_l")
        vars_rest_right = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="robot_r")
        vars_rest = vars_rest_left + vars_rest_right
        saver = tf.train.Saver(vars_rest)
        saver.restore(sess, ckpt)
        print("Model robot restored from file: %s" % ckpt)



    def predict(self, sess, kine):
        kine_l = kine[:,:3]
        kine_r = kine[:,3:]
        feed_dict = {self.kine_l: kine_l, self.kine_r: kine_r, self.is_training:False}
        pred_mask = sess.run(self.pred_mask, feed_dict=feed_dict)
        return np.round(pred_mask)
 

class Trainer(object):
    def __init__(self, data_providers, graph_valid, optimizer="adam", opt_kwargs={}):
        self.net = model()
        self.optimizer_name = optimizer
        self.opt_kwargs = opt_kwargs
        self.data_provider, self.data_provider_valid = data_providers
        self.next, self.init_iter = self.data_provider.get_sample()
        mask_left, mask_right, kine = self.next
        self.net.create_model(mask_left, mask_right, kine[:,:3], kine[:,3:], True)

        self.graph_valid = graph_valid
        with self.graph_valid.as_default() as g:
            self.next_valid, self.init_iter_valid = self.data_provider_valid.get_sample()
            self.net_valid = model()
            mask_left, mask_right, kine = self.next_valid
            self.net_valid.create_model(mask_left, mask_right, kine[:,:3], kine[:,3:], False)
            self.sess_valid = tf.Session(graph=self.graph_valid)


    def _get_optimization(self):
        if self.optimizer_name == "adam":
            learning_rate = self.opt_kwargs.get("learning_rate_adam")
            beta1 = self.opt_kwargs.pop("beta1_adam")
            learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
            #optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(tf.train.AdamOptimizer(learning_rate=learning_rate_node, beta1 = beta1))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node, beta1 = beta1)
            vars_left = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="robot_l")
            vars_train = vars_left 
            update_ops_left = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "robot_l")
            update_ops_right = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "robot_r")
            update_ops = update_ops_left+update_ops_right
            with tf.control_dependencies(update_ops):
                train_step = optimizer.minimize(self.net.cost_l, var_list=vars_train)
            return train_step

    def _initialize(self):
        #initializer for training summaries and evaluation utilities and global variables
        self.train_step = self._get_optimization()

    def _get_summaries(self):
        summaries = []
        with tf.name_scope("training"):
            with tf.name_scope("cost"):
                summaries.append(tf.summary.scalar('left', self.net.cost_l))

            with tf.name_scope("metrics"):
                summaries.append(tf.summary.scalar('f1', tf.reduce_mean(self.net.f1)))
        self.summaries_train = tf.summary.merge(summaries)


    def get_train_summaries_and_reinitialize(self, global_epoch):                
        summary = tf.Summary()
        summary.value.add(tag="training_epoch/loss_l", simple_value=np.mean(self.loss_train_left))

        summary.value.add(tag="training_epoch/f1", simple_value=np.mean(self.f1_train))
        self.summary_writer.add_summary(summary, global_epoch)
        self.loss_train_left = []
        self.f1_train = []

    def train_funct(self, sess, global_iter, global_epoch):
        if not global_iter % self.display_step == 0:
            _, loss_l, f1, summary = sess.run([self.train_step, self.net.cost_l, self.net.f1, self.summaries_train])
            
            self.summary_writer.add_summary(summary, global_iter)

            self.loss_train_left.append(loss_l) 
            self.f1_train.append(np.mean(f1))        

        else:
            _, loss_l, f1, summary, gt_mask, pred_mask = sess.run([self.train_step, self.net.cost_l, self.net.f1, self.summaries_train, self.net.mask, self.net.pred_mask])

            self.summary_writer.add_summary(summary, global_iter)

            self.loss_train_left.append(loss_l) 
            self.f1_train.append(np.mean(f1))      
        
            pred_mask_ = [sigm2image(pred_mask[0])]
            gt_mask_ = [sigm2image(gt_mask[0])]

            save_images((pred_mask_, gt_mask_, [f1[0]]), self.prediction_path_train, global_iter,"train")
            print("Iteration {} of epoch {} - f1: {}".format(global_iter, global_epoch, np.mean(f1)))

        global_iter += 1
        return global_iter





    def train(self, outputs_path_rooth, display_step=500,restore=False, model_to_load_path = None, **kwargs):


        if not os.path.isdir(outputs_path_rooth):
            os.makedirs(outputs_path_rooth)

        prediction_path = "predictions/"
        self.prediction_path_train = os.path.join(outputs_path_rooth,prediction_path,"train")
        os.system("rm -r {}".format(self.prediction_path_train))
        os.makedirs(self.prediction_path_train)
        self.prediction_path_valid = os.path.join(outputs_path_rooth,prediction_path,"valid")
        os.system("rm -r {}".format(self.prediction_path_valid))
        os.makedirs(self.prediction_path_valid)

        logs_path = "logs"
        logs_path = os.path.join(outputs_path_rooth,logs_path)
        os.system("rm -r {}".format(logs_path))
        os.makedirs(logs_path)

        model_path = os.path.join(outputs_path_rooth, "model_ckpts_temp/")
        os.system("rm -r {}".format(model_path))
        os.makedirs(model_path) 
        self.model_path = model_path


 
        self._initialize()
        self.display_step = display_step

        total_epochs = self.data_provider.epochs
        print("Starting optimization:{}-total epochs".format(total_epochs))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #log_device_placement
        self.loop = 0

        initialize_variables = tf.global_variables_initializer()
        self._get_summaries()
        self.saver = tf.train.Saver(max_to_keep = 10)

        nvtx_callback = NVTXHook(skip_n_steps=1, name='Train')
        "with tf.Session(config=config,  graph=tf.get_default_graph()) as sess:"
        with tf.train.MonitoredSession(hooks=[nvtx_callback]) as sess:
            self.summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

            sess.run(initialize_variables)

            sess.run(self.init_iter)

            if restore:
                ckpt = tf.train.get_checkpoint_state(model_to_load_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(ckpt.model_checkpoint_path, sess)

            print("Start optimization")
            global_epoch = 0
            global_iter = 0


            for epoch in range(total_epochs):
                self.loss_train_left = []
                self.f1_train = []
                s = time.time()
                i_c = 0
                times=[]
                while True:
                    try:
                        s = time.time()
                        global_iter = self.train_funct(sess, global_iter, global_epoch + 1)
                        times.append(time.time()-s)
                        if i_c % 100:
                            print(np.mean(times))
                        i_c += 1
                    except tf.errors.OutOfRangeError:
                        t_epoch = time.time() - s
                        print("epoch run in {:.2f} ({:.5f}s/iter)".format(t_epoch, t_epoch/i_c))
                        exit()
                        global_epoch += 1
                        self.net.save(self.saver, sess, model_path, global_epoch)
                        self.get_train_summaries_and_reinitialize(global_epoch)
                        self.run_validation(sess, global_epoch)
                        sess.run(self.init_iter)
                        break
                    
        print("\n\nOptimization Finished!\n\n")


    def run_validation(self, sess, global_epoch):
        print("Running validation")
        pm_v = []
        gt_v = []
        i_v = 0
        f1_v = []
        with self.graph_valid.as_default() as g:
            with tf.Session(graph=g) as sess:
                self.net_valid.restore_inference(tf.train.get_checkpoint_state(self.model_path).model_checkpoint_path, sess) 

                print("Running validation")
                sess.run(self.init_iter_valid)
                loss_l_v = []
                loss_r_v = []
                f1_v = []
                f1_save_v = []
                pred_mask_v = []
                mask_gt_v = []
                i = 0

                while True:
                    try:   
                        if i % 4 == 0:
                            loss_l, loss_r, f1, mask_gt, pred_mask = sess.run([self.net_valid.cost_l, self.net_valid.cost_r, 
                                self.net_valid.f1, self.net_valid.mask, self.net_valid.pred_mask])

                            pred_mask_v.append(sigm2image(pred_mask[0]))
                            mask_gt_v.append(sigm2image(mask_gt[0]))
                            f1_save_v.append(f1[0])
                            loss_l_v.append(loss_l)
                            loss_r_v.append(loss_r)
                            f1_v.extend(f1) 

                        else:
                            loss_l, loss_r, f1 = sess.run([self.net_valid.cost_l, self.net_valid.cost_r, 
                                self.net_valid.f1])
                            loss_l_v.append(loss_l)
                            loss_r_v.append(loss_r)
                            f1_v.extend(f1) 

                        i += 1
                        

                    except tf.errors.OutOfRangeError:
                        break

                save_images((pred_mask_v, mask_gt_v, f1_v), self.prediction_path_valid, global_epoch,"valid")

                av_loss_l = np.mean(loss_l_v)
                av_loss_r = np.mean(loss_r_v)
                av_f1 = np.mean(f1_v)

                summary = tf.Summary()
                summary.value.add(tag="validation/loss_l", simple_value=av_loss_l)
                summary.value.add(tag="validation/loss_r", simple_value=av_loss_r)
                summary.value.add(tag="validation/f1", simple_value=av_f1)
                self.summary_writer.add_summary(summary, global_epoch)

                print("Validation of epoch {} -  f1: {}".format(global_epoch, av_f1))

        
class Tester(object):
    def __init__(self, net, ckpts_path, save_path, data_provider):
        self.net = net
        ckpt = tf.train.get_checkpoint_state(ckpts_path).model_checkpoint_path
        self.sess = net.restore(ckpt, mode = "test")
        self.save_path = save_path
        if not os.path.isdir(save_path): os.makedirs(save_path)
        self.data_provider = data_provider
        self.next_sample, self.init_iter = data_provider.get_sample()

    def test(self):
        sess = self.sess
        sess.run(self.init_iter) 
        global_index = 0
        i_v = 0
        f1_v = []
        while True:
            try:
                mask, kine = sess.run(self.next_sample)
                if (i_v+1) % 1 == 0:
                    kine_l = kine[:,:3]
                    kine_r = kine[:,3:]

                    kine_l, kine_r = get_random_kine()

                    feed_dict = {self.net.kine_l: kine_l, self.net.kine_r: kine_r, self.net.mask_l: mask, self.net.mask_r: np.zeros(mask.shape), self.net.is_training:False}
                    print(i_v, kine_l, kine_r)
                    f1, pred_mask, gt_mask = sess.run([self.net.f1, self.net.pred_mask, self.net.mask], feed_dict=feed_dict)

                    save_test([sigm2image(pred_mask[0]), sigm2image(gt_mask[0])], self.save_path, i_v, f1)
                    #self.augment_and_save(sess, [pred_mask[0], gt_mask[0], kine_l[0], kine_r[0]],i_v)
                    
                    f1_v.append(f1)

     
                #if i_v == 50:
                    #exit()
                i_v += 1

            except tf.errors.OutOfRangeError:
                break
        print("finished: {} elements, average f1: {}".format(i_v, np.mean(f1_v)))


    def augment_and_save(self, sess, predictions, index):
        path = self.save_path
        if index == 0:
            os.system("rm -r {}".format(path))
            os.makedirs(path)
        pred_mask, gt_mask, kine_l, kine_r = predictions
        kines = [kine_l, kine_r]
        pm0 = Image.fromarray(np.uint8(np.squeeze(sigm2image(pred_mask)))).convert('P') 
        
        for i_arm, arm in enumerate(["left","right"]):
            kine_rooth = np.copy(kines[i_arm])
            kine_fixed = np.expand_dims(np.copy(kines[1-i_arm]), axis=0)
            for i_q, q in enumerate(["transl","rot","bend"]):
                pmi_v = [pm0]
                for i in range(10):
                    k_plus = kine_rooth[i_q]+(i+1)/50
                    print(k_plus)
                    k_modified = np.copy(kine_rooth)
                    k_modified[i_q] = k_plus
                    k_increased = np.expand_dims(k_modified, axis=0)
                    k_r = np.expand_dims(kine_r, axis=0)
                    if i_arm == 0:
                        feed_dict = {self.net.kine_l: k_increased, self.net.kine_r: kine_fixed, self.net.is_training:False}
                    else:
                        feed_dict = {self.net.kine_l: kine_fixed, self.net.kine_r: k_increased, self.net.is_training:False}
                    pred_mask = sess.run(self.net.pred_mask, feed_dict=feed_dict)
                    pmi = Image.fromarray(np.uint8(np.squeeze(sigm2image(pred_mask[0])))).convert('P') 
                    pmi_v.append(pmi)
                pmi_v[0].save(os.path.join(path + "{}_{}_{}.gif".format(index,arm,q)), save_all=True, append_images=pmi_v[1:], optimize=False, duration=150, loop=0)
        if index == 10:
            exit()





    
    
