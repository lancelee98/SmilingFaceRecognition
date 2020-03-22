import os
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
tf.disable_v2_behavior()


# 读取图片信息，存储到 tfrecord格式文件
def read_image_to_tfrecode(path, save_name):
    source_file_path = path
    classes = []
    for dirname in os.listdir(source_file_path):
        classes.append(dirname)
    writer = tf.python_io.TFRecordWriter(save_name)
    # 为了保证标签与对应的数字不会混乱，我们对类别集合按自然排序的方法进行排序，这样在制作训练数据和
    # 测试数据时实现彼此中类别与数字对应关系是一样的
    classes.sort()
    for index, name in enumerate(classes):
        class_path = source_file_path + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img_raw = img.tobytes()
            if index == 0:
                label_val = np.array([1., 0.])
            else:
                label_val = np.array([0., 1.])
            label_val = label_val.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_val])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


# read_image_to_tfrecode('./train-jpg/')
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)

    image = tf.reshape(image, [32, 20, 1])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    l = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(tf.cast(l, tf.float32), [2])
    return image, label


def get_tfrecord(batch_size, tf_record_path):
    image, lable = read_and_decode(tf_record_path)
    image_batch, lable_batch = tf.train.shuffle_batch([image, lable], batch_size=batch_size,
                                                      num_threads=2, capacity=400,
                                                      min_after_dequeue=200)  # 制作每次喂入神经网络的数据量，取决于batch_size
    return image_batch, lable_batch

# read_image_to_tfrecode('C:/MTFL/train/', 'smile_train.tfrecords')
# image_batch, lable_batch = get_tfrecord(200, 'smile_test.tfrecords')
# # image_batch,lable_batch=read_and_decode('smile_train.tfrecords')
# x = tf.placeholder(tf.float32, [None, 32, 20, 1])
# y_ = tf.placeholder(tf.int32, [None, 2])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     threads = tf.train.start_queue_runners(sess=sess)
#     b_image, b_lable = sess.run([image_batch, lable_batch])
#     print(sess.run([y_], feed_dict={x: b_image, y_: b_lable}))
# sess.close()
