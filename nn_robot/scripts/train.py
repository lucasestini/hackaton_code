import sys
import os
root_dir = "/p/home/jusers/sestini1/juwels/shared/hackaton_code/" #change
sys.path.insert(0, os.path.join(root_dir,"nn_robot/model/"))
import whole_model
from data_provider_train import data_provider as data_provider_train
from data_provider_test import data_provider as data_provider_test
import tensorflow as tf 
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if "GPU" in str(x.device_type)])

if __name__ == '__main__':

    if not tf.test.is_gpu_available():
        #raise
        pass
    print(tf.__version__)
    print("Available GPUs:\n")
    get_available_gpus()
    print("\n\n\n")

    dataset_train = os.path.join(root_dir,"data/tfr_kine_dataset/train/")
    dataset_test = os.path.join(root_dir,"data/tfr_kine_dataset/test/")

    batch_size = 128
    valid_size = 64

    buffer_size_dataset = 500

    epochs = 500
    augment = False

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs")
        exit()
    gpus_number = len(gpus)
    
    devices = ["/GPU:"+str(i) for i in range(gpus_number)]
    print("\n{} devices: \n{}\n".format(gpus_number, gpus))



    data_generator_train = data_provider_train(dataset_train, epochs, batch_size, buffer_size_dataset, augment)

    graph_valid = tf.Graph()
    with graph_valid.as_default():
        data_generator_valid = data_provider_test(dataset_test, valid_size,)
    print("---Generator Built")

    optimizer = "adam"
    

    opt_args = {"learning_rate_adam": 5e-3, "beta1_adam":0.9}
    
    trainer = whole_model.Trainer([data_generator_train, data_generator_valid], graph_valid, optimizer=optimizer, opt_kwargs=opt_args)
    print("---Trainer Built")

    outputs_path_rooth = os.path.join(root_dir,"nn_robot/results/results_1/")
    restore = False
    model_to_load_path = None
    trainer.train(outputs_path_rooth, display_step=100, restore=restore, model_to_load_path=model_to_load_path)



