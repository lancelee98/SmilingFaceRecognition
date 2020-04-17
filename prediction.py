import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import utils.tfdata_tools as tt

tf.disable_v2_behavior()
x_image = tf.placeholder(tf.float32, [None, 32, 20, 1])  # 32*20
y_ = tf.placeholder(tf.float32, [None, 2])  # 0 1 0号位置为1代表不笑 1号位置为1代表笑

# train capacity=10000 smile=5616 not_smile=4384
image_total, label_total = tt.get_tfrecord(10000, '..//data/firstGround//train//smile_train.tfrecords', capacity=10000)
# test capacity=2995 smile=1632 not_smile=1363
image_batch_test, label_batch_test = tt.get_tfrecord(2995, '..//data/firstGround//test//smile_test.tfrecords', capacity=2995)

def predict():

    # with tf.Session() as sess:
    #     b_image_test, b_label_test = sess.run([image_batch_test, label_batch_test])
    #     b_image_total, b_label_total = sess.run([image_total, label_total])
    #
    #     new_saver = tf.train.import_meta_graph('D:\PythonProject\SmilingFaceRecognition\models\model/third_ground_model.ckpt.meta')
    #     new_saver.restore(sess, 'D:\PythonProject\SmilingFaceRecognition\models\model/third_ground_model.ckpt.data-00000-of-00001')
    #     image = tf.reshape(img, [1, 32, 20, 1])
    #     image = sess.run(image)
    #     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #     label_val = np.array([1., 0.])
    #     # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    #     y_conv = tf.get_collection('y_conv')[0]
    #     accuracy = tf.get_collection('accuracy')[0]
    #
    #     graph = tf.get_default_graph()
    #
    #     # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
    #     x_image = graph.get_operation_by_name('x_image').outputs[0]
    #     y_ = graph.get_operation_by_name("y_").outputs[0]
    #
    #     # 使用y进行预测
    #     sess.run(accuracy, feed_dict={x_image: image, y_: label_val})

    # img = Image.open(imgpath)
    saver = tf.train.import_meta_graph('D:\PythonProject\SmilingFaceRecognition\models\model/first_ground_model.ckpt.meta')
    gragh = tf.get_default_graph()
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
    print(tensor_name_list)
    # with tf.Session() as sess:
    #     sava_model = tf.train.latest_checkpoint(')
    #     saver.restore(sess, sava_model)
    #     b_image_test, b_label_test = sess.run([image_batch_test, label_batch_test])
    #     b_image_total, b_label_total = sess.run([image_total, label_total])
    #     # image = tf.reshape(img, [1, 32, 20, 1])
    #     # image = sess.run(image)
    #     # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #     # 获取权重
    #     graph = tf.get_default_graph()  # 获取当前默认计算图
    #     print(graph)
    #     accuracy = graph.get_collection("accuracy")
    #     x_image = graph.get_collection("x_image")
    #     y_ = graph.get_collection("y_")
    #     y_conv = graph.get_collection("y_conv")  # get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
    #     train_a = accuracy.eval(feed_dict={x_image: b_image_total, y_: b_label_total})
    #     print(train_a)
