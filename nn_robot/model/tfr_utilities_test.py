import os
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from PIL import Image

# -----------------------------  WRITING -----------------------------------------------------------

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _floats_list_feature(lst):
    return tf.train.FeatureList(feature = [tf.train.Feature(float_list=tf.train.FloatList(value=[val])) for val in lst])

def write_tfrecords_file(out_path, samples, samples_per_file, n_records):
    print("... Building records at {}".format(out_path))
    file_rooth = os.path.join(out_path,"tf_record_{}.tfrecords")
    start_read = 0
    for i_file in range(n_records):
        with tf.io.TFRecordWriter(file_rooth.format(i_file)) as writer:
            for i in range(samples_per_file):
                f, b, k = samples[start_read + i]
                f_bytes = open(f,'rb').read()
                b_bytes = open(b,'rb').read()

                data = {'frame': _bytes_feature(f_bytes), 
                        'back': _bytes_feature(fp_bytes),
                        "kine": _floats_feature(k)}

                feature = tf.train.Features(feature=data)  # Wrap the data as TensorFlow Features.
                example = tf.train.Example(features=feature)  # Wrap again as a TensorFlow Example.
                serialized = example.SerializeToString()  # Serialize the data.
                writer.write(serialized)  # Write the serialized data to the TFRecords file.
                                
                if (start_read + i) == (len(samples) - 1):
                    break

        start_read += i + 1
        if (i_file+1) % 5 == 0: 
            print("Record {}/{} saved".format(i_file+1, n_records))




# -----------------------------  READING -----------------------------------------------------------

def parse(serialized, shape=(570, 760, 3), shape_mask=(570, 760, 1)):
    features = {
        "left": tf.io.FixedLenFeature([], tf.string),
        "right": tf.io.FixedLenFeature([], tf.string),
        "kine": tf.io.FixedLenFeature([6], tf.float32)
    }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)


    ml = parsed_example['left']  # Get the image as raw bytes.
    ml = tf.image.decode_image(ml)  # Decode the raw bytes so it becomes a tensor with type.
    mr = parsed_example['right']  # Get the image as raw bytes.
    mr = tf.image.decode_image(mr)  # Decode the raw bytes so it becomes a tensor with type.
    k = parsed_example["kine"]
    ml = tf.cast(tf.reshape(ml, shape=shape_mask), dtype=tf.float32)/255.
    mr = tf.cast(tf.reshape(mr, shape=shape_mask), dtype=tf.float32)/255.

    return ml, mr, k


def read_tfrs_augm(tf_v, batch_size, buffer_size, AUTOTUNE, augment, preprocess_f=None):
    dataset = tf_v.interleave(lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=os.cpu_count()),
                              block_length=buffer_size, cycle_length=100,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(parse,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # parse tfrecords. Parameter num_parallel_calls may help performance.

    if augment:
        dataset = dataset.map(AugmentDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not preprocess_f is None:
        dataset = dataset.map(lambda a, b, c: (a,b, c),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def AugmentDataset(f, m, k):
    exit()
    coin = tf.less(tf.random_uniform((), 0., 1.), 0.5)
    f = tf.cond(coin, lambda: tf.image.flip_left_right(f), lambda: f)
    m = tf.cond(coin, lambda: tf.image.flip_left_right(m), lambda: m)
    return f, m, k



def prepare_dataset(dataset_path, batch_size, buffer_size, augment=True, preprocess_f=None):
    tfr = os.path.join(dataset_path, "*.tfrecords")
    tf_v = tf.data.Dataset.list_files(tfr).shuffle(100)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = read_tfrs_augm(tf_v, batch_size, buffer_size, AUTOTUNE, augment, preprocess_f)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    init_op = iterator.initializer  # can be run to reinitialize the dataset

    return next_element, init_op



