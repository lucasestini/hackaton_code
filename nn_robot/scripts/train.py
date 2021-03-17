import sys
sys.path.insert(0, "/home/lucas/nn_robot_new/model/")
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

    dataset_train = "/home/lucas/tfr_new_dataset_kine/tfr_new_dataset_kine_train/"
    dataset_valid = "/home/lucas/tfr_new_dataset_kine/tfr_new_dataset_kine_test/"
    batch_size = 32
    valid_size = 16

    buffer_size_dataset = 200

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
        data_generator_valid = data_provider_test(dataset_valid, valid_size,)
    print("---Generator Built")

    optimizer = "adam"
    

    opt_args = {"learning_rate_adam": 5e-3, "beta1_adam":0.9}
    
    trainer = whole_model.Trainer([data_generator_train, data_generator_valid], graph_valid, optimizer=optimizer, opt_kwargs=opt_args)
    print("---Trainer Built")

    outputs_path_rooth = "/home/lucas/nn_robot_new/results/results_1/"
    restore = False
    model_to_load_path = None
    trainer.train(outputs_path_rooth, display_step=5, restore=restore, model_to_load_path=model_to_load_path)



