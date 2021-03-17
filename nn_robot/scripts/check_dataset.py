import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0,'/home/lucas/atlas_work/code/vtk_robot_model/vtk_robot/project_module/')
sys.path.insert(0,'/home/lucas/atlas_work/code/regression/nn_robot/model/')
from project_main import project_dataset
from util import combine_left_right, unnormalize_kine, load_mm_list
import time
from tfr_utilities import prepare_dataset
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


tmp_fold = "/home/lucas/atlas_work/code/regression/nn_robot/results/tmp/"

dataset_rooth = "/home/lucas/atlas_work/datasets/tfr_kinematics/train/"
next_sample, init = prepare_dataset(dataset_rooth, batch_size=32, buffer_size=500, augment=False)
tmp_fold = os.path.join(tmp_fold, "tfr_kinematics_examples")
os.system("rm -r {}".format(tmp_fold));  os.makedirs(tmp_fold)

mm_list = load_mm_list("/home/lucas/atlas_work/code/vtk_robot_model/vtk_robot/")
with tf.Session() as sess:
    sess.run(init)
    s = time.time()
    i_c = 0
    i_error = 0
    while True:
        try:
            m, k = sess.run(next_sample)
            for i in range(32):
                try:
                    print(m[0].max(), m[0].min(), len(np.unique(m[0])))
                    print(aaaaaaaaaaaaaaaaa)
                    mi = np.repeat(np.uint8(m[i]), axis=2, repeats=3)
                    k_u = unnormalize_kine(k[i], mm_list)
                    k_feed = [k_u[:3], k_u[3:]]
                    img_project_l, img_project_r, _ = project_dataset(k_feed)
                    proj_k = combine_left_right(img_project_l, img_project_r)
                    ki = np.repeat(np.expand_dims(np.uint8(proj_k * 255), axis=2), repeats=3, axis=2)

                    concat = np.concatenate([mi,ki], axis = 1)
                    Image.fromarray(concat).save(os.path.join(tmp_fold,"{}.jpg".format(i_c)))

                    i_c += 1
                    if i_c == 20:
                        exit()
                except IndexError:
                    i_error += 1


        except tf.errors.OutOfRangeError:
            elaps = time.time() - s
            print(
                "Finished: {} samples read in {}s ({:.2}s/sample), {}perc error".format(i_c * 32, elaps, elaps / i / 32,
                                                                                        i_error / (i_c * 32) * 100))
            break

