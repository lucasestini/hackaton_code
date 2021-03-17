import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import psutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pickle as pkl
import time


class MyException(Exception):
    pass

def clip(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def sigm2image(sigm):
    return sigm*255.

def tanh2image(tanh):
    return tanh*127.5 + 127.5

def image2tanh(image):
    return (image - 127.5)/127.5

def tanh2caffe(tanh):
    im = tanh2image(tanh)
    return image2caffe_vgg(im)

def sigm2caffe(sigm):
    im = sigm2image(sigm)
    return image2caffe_vgg(im)

def image2caffe_vgg(image):
    image = image[...,::-1] #convert to bgr
    mean_channels = tf.reshape(tf.constant([103.939, 116.779, 123.68], tf.float32), [1,1,1,3])
    return image - mean_channels

def mask2caffe(tanh):
    im = sigm2image(tanh)
    im = tf.repeat(im, repeats = 3, axis = 3)
    return image2caffe_vgg(im)

def randomize(im):
    shape = tf.shape(im)[0]
    angles = tf.random.uniform([shape], minval=-10, maxval=10)*tf.math.pi/180.
    return tfa.image.rotate(tf.image.random_flip_left_right(tf.image.random_flip_up_down(im)), angles=angles, interpolation="BILINEAR")

def randomize(im):
    shape = tf.shape(im)[0]
    angles = tf.random.uniform([shape], minval=-10, maxval=10)
    return tfa.image.rotate(tf.image.random_flip_left_right(tf.image.random_flip_up_down(im)), angles=angles, interpolation="BILINEAR")


def save_images(predictions, prediction_path, global_epoch, mode):
    pred_mask, gt_mask, f1 = predictions
    if mode == "valid":
        prediction_path = os.path.join(prediction_path, str(global_epoch))
        if not os.path.isdir(prediction_path):
            os.makedirs(prediction_path)
            n_chars = len(str(len(pred_mask)))
    else:
        n_chars = 10
    for i in range(len(pred_mask)):
        if mode == "valid":
            f_name = str(i)
        else:
            f_name = str(global_epoch)
        while len(f_name) < n_chars:
            f_name = "0" + f_name
        f_name = f_name + ".jpg"


        pm = np.uint8(pred_mask[i])
        gt = np.uint8(gt_mask[i])

        overlap = np.uint8(plot_diff(pm, gt)*255.)

        conc = np.squeeze(np.concatenate([gt, pm, overlap], axis = 1))

        img = Image.fromarray(conc)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/p/home/jusers/sestini1/juwels/shared/hackaton_code/data/fonts/NimbusSanL-Bol.otf", 35)
        i_c = 0
        captions = ["Ground Truth", "Predicted Mask", "Overlap: {:.2f}%".format(f1[i]*100)]
        for w in range(1):
            for h in range(3):
                draw.text((20 + pm.shape[1]*h, 20 + pm.shape[0]*w), captions[i_c], 255, font = font)
                i_c += 1

        img.save(os.path.join(prediction_path,f_name))

def save_images_valid(predictions, prediction_path, global_epoch, mode):
    pred_mask, gt_mask = predictions
    if mode == "valid":
        prediction_path = os.path.join(prediction_path, str(global_epoch))
        if not os.path.isdir(prediction_path):
            os.makedirs(prediction_path)
        n_chars = len(str(len(pred_mask)))
    else:
        n_chars = 10
    for i in range(len(pred_mask)):
        if mode == "valid":
            f_name = str(i)
        else:
            f_name = str(global_epoch)
        while len(f_name) < n_chars:
            f_name = "0" + f_name
        f_name = f_name + ".jpg"

        pm = np.uint8(pred_mask[i])
        gt = np.uint8(gt_mask[i])
        overlap = np.uint8(plot_diff(pm, gt)*255.)
        
        conc = np.squeeze(np.concatenate([gt, pm, overlap], axis = 1))

        img = Image.fromarray(conc)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/home/lsestini/atlas_work/data/other_resources/nimbus-sans-l/NimbusSanL-Bol.otf", 35)
        i_c = 0
        captions = ["Ground Truth", "Predicted Mask", "Overlap"]
        for w in range(1):
            for h in range(3):
                draw.text((20 + pm.shape[1]*h, 20 + pm.shape[0]*w), captions[i_c], 255, font = font)
                i_c += 1

        img.save(os.path.join(prediction_path,f_name))


def plot_diff(pred_, gt_):
    pred = np.uint8(np.round(pred_/255.)*255)
    gt = np.uint8(np.round(gt_/255.)*255)
    shape = np.array(pred).shape
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if pred[i,j] == 255 and gt[i,j] == 0:
                out[i,j] = 0.5
            elif pred[i,j] == 0 and gt[i,j] == 255:
                out[i,j] = 1.
    return out

def get_memory():
    return psutil.virtual_memory().percent

def show(np_im):
    Image.fromarray(np.uint8(np_im)).show()

def str2number(st):
    while True:
        char = st[0]
        if char != "0":
            return int(st)
        else:
            if len(st) == 1:
                return int(st)
            else:
                st = st[1:]
    return int(s)



def normalize(data,mm):
    max_ = mm[0]
    min_ = mm[1]
    return (data - min_)/(max_ - min_)

def normalize_kine(kine_v,mm):
    assert kine.shape[1] == len(mm)
    for i in range(kine.shape[1]):
        kine[:,i] = normalize(kine[:,i], mm[i])
    return kine

def unnormalize(kine,mm):
    max_ = mm[0]
    min_ = mm[1]
    return kine*(max_ - min_) + min_

def unnormalize_kine(kine,mm):
    kine_out = []
    for i in range(6):
        kine_out.append(unnormalize(kine[i], mm[i]))
    return kine_out

def combine_left_right(im1, im2):
    return (im1 + im2 > 0).astype("int")

def get_file_name(i,n_chars):
    name = str(i)
    while len(name) < n_chars:
        name = "0" + name
    return name

def load_mm_list(my_vtk_folder):
    with open(my_vtk_folder + "/project_utilities/max_min_joints.pkl", "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        mm = u.load()
        mm_list = []
        for i in range(2):
            for j in range(3):
                mm_list.append(mm[i][j])
    return mm_list

def project(k, mm_list):
    k_u = unnormalize_kine(k,mm_list)
    k_feed = [k_u[:3], k_u[3:]]
    s = time.time()
    img_project_l, img_project_r, _ = project_dataset(k_feed)
    print(time.time() - s)
    mask = np.expand_dims(combine_left_right(img_project_l, img_project_r), axis = 2)
    return np.expand_dims(mask, axis = 0)


def augment(mask, kine):
    mm = load_mm_list('/home/lucas/atlas_work/code/vtk_robot_model/vtk_robot/')
    for i,k in enumerate(kine):
        print(k)
        noise = np.random.normal(0,0.05, size = k.shape)
        k_noised = np.clip(k + noise, a_min = 0., a_max = 1.)
        mask_noised = project(k_noised, mm)
        mask = np.concatenate([mask, mask_noised], axis = 0)
        kine = np.concatenate([kine, np.expand_dims(k_noised, axis = 0)], axis = 0)
        try:
            mask_noised = project(k_noised, mm)
            if i == 0:
                mask_out = np.copy(mask_noised)
                kine_out = np.expand_dims(k_noised, axis = 0)
            else:
                mask_out = np.concatenate([mask_out, mask_noised], axis = 0)
                kine_out = np.concatenate([kine_out, np.expand_dims(k_noised, axis = 0)], axis = 0)
        except IndexError:
            pass

    return mask_out, kine_out

def save_test(predictions, prediction_path, global_epoch, f1):
    pred_mask, gt = predictions
    prediction_path = os.path.join(prediction_path, "paired")
    if not os.path.isdir(prediction_path):
        os.makedirs(prediction_path)
    n_chars = 4
    f_name = str(global_epoch)
    while len(f_name) < n_chars:
        f_name = "0" + f_name
    f_name = f_name + ".jpg"

    pm = np.uint8(pred_mask)
    gt = np.uint8(gt)
    overlap = np.uint8(plot_diff(pm, gt)*255.)

    conc = np.concatenate([gt, pm, overlap], axis = 1)

    img = Image.fromarray(np.squeeze(conc))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/home/lsestini/atlas_work/data/other_resources/nimbus-sans-l/NimbusSanL-Bol.otf", 35)
    i_c = 0
    captions = ["Ground Truth", "Predicted Mask", "Overlap: {:.2f} f1-score".format(f1*100)]
    for w in range(1):
        for h in range(3):
            draw.text((20 + pm.shape[1]*h, 20 + pm.shape[0]*w), captions[i_c], 255, font = font)
            i_c += 1

    img.save(os.path.join(prediction_path,f_name))


"""def get_random_kine():
    tl = np.random.normal(0.7,0.1/2.)
    rl = np.random.normal(0.5,0.2/2.)
    bl = np.random.normal(0.2,0.05/2.)
    tr = np.random.normal(0.75,0.05/2.)
    rr = np.random.normal(0.5,0.1/2.)
    br = np.random.normal(0.3,0.1/2.)

    kl = np.expand_dims([tl,rl,bl], axis=0)
    kr = np.expand_dims([tr,rr,br], axis=0)
    return kl, kr"""

def get_random_kine():
    tl = np.random.normal(0.7,0.1/2.)
    rl = np.random.normal(0.5,0.1/2.)
    bl = np.random.normal(0.2,0.05/2.)
    tr = np.random.normal(0.75,0.05/2.)
    rr = np.random.normal(0.5,0.1/2.)
    br = np.random.normal(0.8,0.1/2.)

    kl = np.expand_dims([tl,rl,bl], axis=0)
    kr = np.expand_dims([tr,rr,br], axis=0)
    return kl, kr
