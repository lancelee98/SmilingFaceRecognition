import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np

tf.disable_v2_behavior()
x_image = tf.placeholder(tf.float32, [None, 32, 20, 1])  # 32*20
y_ = tf.placeholder(tf.float32, [None, 2])  # 0 1 0号位置为1代表不笑 1号位置为1代表笑

def predict(imgpath):
    img = Image.open(imgpath)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('D:\PythonProject\SmilingFaceRecognition\models\model/third_ground_model.ckpt.meta')
        new_saver.restore(sess, 'D:\PythonProject\SmilingFaceRecognition\models\model/third_ground_model.ckpt.data-00000-of-00001')
        image = tf.reshape(img, [1, 32, 20, 1])
        image = sess.run(image)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        label_val = np.array([1., 0.])
        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        y_conv = tf.get_collection('y_conv')[0]
        accuracy = tf.get_collection('accuracy')[0]

        graph = tf.get_default_graph()

        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        x_image = graph.get_operation_by_name('x_image').outputs[0]
        y_ = graph.get_operation_by_name("y_").outputs[0]

        # 使用y进行预测
        sess.run(accuracy, feed_dict={x_image: image, y_: label_val})


    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     sava_model = tf.train.latest_checkpoint('D:\PythonProject\SmilingFaceRecognition\models\model')
    #     saver.restore(sess, sava_model)
    #     image = tf.reshape(img, [1, 32, 20, 1])
    #     image = sess.run(image)
    #     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #     # 获取权重
    #     graph = tf.get_default_graph()  # 获取当前默认计算图
    #     print(graph)
    #     y_conv = graph.get_collection("y_conv")  # get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
    #     x_image = graph.get_collection("x_image")
    #     y_ = graph.get_collection("y_")
    #     accuracy = graph.get_collection("accuracy")
    #     label_val = np.array([1., 0.])
    #     sess.run(accuracy, feed_dict={x_image: image, y_: label_val})
    #     print(accuracy)
