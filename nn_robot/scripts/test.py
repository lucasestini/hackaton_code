import sys
sys.path.insert(0, "/home/lsestini/atlas_work/code/nn_robot/model/")
import whole_model
from data_provider_one_only import data_provider
from data_provider_test import data_provider as data_provider_test
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

    dataset_rooth = "/home/lsestini/atlas_work/data/tfr_kinematics_lr/"
    dataset_rooth2 = "/home/lsestini/atlas_work/data/tfr_kinematics_augm_lr/"
    dataset_valid = "/home/lsestini/atlas_work/data/tfr_kinematics/test/"
    batch_size = 64
    valid_size = 64
    test_size = 1

    buffer_size_dataset = 20 #1000

    epochs = 500
    augment = False
    data_generator = data_provider(dataset_rooth, epochs, batch_size, buffer_size_dataset, augment)
    data_generator2 = data_provider(dataset_rooth2, epochs, batch_size, buffer_size_dataset, augment)
    data_generator_valid = data_provider(dataset_valid, epochs, valid_size, buffer_size_dataset, augment, flag="valid")
    print("---Generator Built")

    robot = whole_model.model([data_generator, data_generator2, data_generator_valid])
    print("---Model Built")

    
    outputs_path_rooth = "/home/lsestini/atlas_work/code/nn_robot/results/results1/predictions/test/manual/"
    model_to_load_path = "/home/lsestini/atlas_work/data/trained_models/nn_robot/model_ckpts_temp"
    dataset_rooth_test = dataset_valid

    data_generator_test = data_provider_test(dataset_rooth_test, test_size)
    tester = whole_model.Tester(robot, model_to_load_path, outputs_path_rooth, data_generator_test)
    tester.test()

    
