import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import utils.tfdata_tools as tt

tf.disable_v2_behavior()
# train_path = 'D:\PythonProject\SmilingFaceRecognition\data\firstGround\train\smile_train.tfrecords'
test_path = 'D:\PythonProject\SmilingFaceRecognition\data/firstGround/test\smile_test.tfrecords'
# train capacity=10000 smile=5616 not_smile=4384
# image_total, label_total = tt.get_tfrecord(10000, train_path, capacity=10000)
# test capacity=2995 smile=1632 not_smile=1363
image_batch_test, label_batch_test = tt.get_tfrecord(2995, test_path, capacity=2995)


# saver = tf.train.import_meta_graph('D:\PythonProject\SmilingFaceRecognition\models\model/first_ground_model.ckpt.meta')
# gragh = tf.get_default_graph()
# tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
# print(tensor_name_list)
# x_image = gragh.get_tensor_by_name('x-input:0')
# y_ = gragh.get_tensor_by_name('y-input:0')
# y_conv = gragh.get_tensor_by_name('layer7-full2/y_conv:0')
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.argmax(y_, 1)))
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 求出最大值的坐标并做对比
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


def predict_pic(imgpath):
    img = Image.open(imgpath)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'D:\PythonProject\SmilingFaceRecognition\models\model/first_ground_model.ckpt.meta')
        gragh = tf.get_default_graph()
        x_image = gragh.get_tensor_by_name('x-input:0')
        y_ = gragh.get_tensor_by_name('y-input:0')
        y_conv = gragh.get_tensor_by_name('layer7-full2/y_conv:0')
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.argmax(y_, 1)))
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 求出最大值的坐标并做对比
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver.restore(sess, tf.train.latest_checkpoint('D:\PythonProject\SmilingFaceRecognition\models\model'))
        print('finish loading model!')
        image = np.array(img) * (1. / 255) - 0.5
        image = image.reshape([1, 32, 20, 1])
        y = np.array([0, 1])
        y = y.reshape(1, 2)
        c, a, y = sess.run([cost, accuracy, y_conv], feed_dict={x_image: image, y_: y})
        print(c, a, np.argmax(y))


def predict_test():
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        b_image_test, b_label_test = sess.run([image_batch_test, label_batch_test])
        coord.request_stop()
        coord.join(threads)

        saver = tf.train.import_meta_graph(
            'D:\PythonProject\SmilingFaceRecognition\models\model/first_ground_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('D:\PythonProject\SmilingFaceRecognition\models\model'))
        gragh = tf.get_default_graph()
        tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
        # print(tensor_name_list)
        x_image = gragh.get_tensor_by_name('x-input:0')
        y_ = gragh.get_tensor_by_name('y-input:0')
        y_conv = gragh.get_tensor_by_name('layer7-full2/y_conv:0')
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.argmax(y_, 1)))
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 求出最大值的坐标并做对比
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        a, c, y = sess.run([accuracy, cost, y_conv], feed_dict={x_image: b_image_test, y_: b_label_test})
        print(a, c)


# predict_pic('D:\PythonProject\SmilingFaceRecognition\data/firstGround/test/1/0002-image04733.jpg')
# predict_test()
