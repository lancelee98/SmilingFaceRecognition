#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import time
import utils.tfdata_tools as tt
import matplotlib.pyplot as plt

learning_rate = 0.0001


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


x_image = tf.placeholder(tf.float32, [None, 32, 20, 1])  # 32*20
y_ = tf.placeholder(tf.float32, [None, 2])  # 0 1 0号位置为1代表不笑 1号位置为1代表笑

# batch_xs, batch_ys = ct.get_tfrecord(5, 'smile_train.tfrecords')
# x_image = tf.reshape(x, [-1]) #-1代表任意维数 32*20*1 高度 宽度 色通道数

# 第一个卷积层
# 初始化卷积核和偏置值 [32,20,1]->[32,20,10]
filter1 = weight_variable([5, 5, 1, 12])  # 卷积核是由5*5大小的卷积，输入为1个通道而输出为10
bias1 = bias_variable([12])  # 生成的偏置值与卷积结果进行求和的计算
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1],
                     padding='SAME')  # strides 每一维的步长 1、4默认1 ，2、3分别代表平行竖直的步长 SAME代表卷积后与原输入一致
h_conv1 = tf.nn.relu(conv1 + bias1)  # 求得第一个卷积层输出结果 32*20 第一次卷积C1

# maxPooling池化层，对于2*2大小的框进行最大特征取值 [32,20,10]->[16,10,10]
maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16*10 第一次池化S2

# 第二个卷积层 [16,10,10]->[16,10,12]
filter2 = weight_variable([5, 5, 12, 24])
bias2 = bias_variable([24])
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')  # 16*10 第二次卷积 C3
h_conv2 = tf.nn.relu(conv2 + bias2)

# [16,10,12]->[5,5,12]
maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 3, 2, 1], padding='VALID')  # 5*5 第二池化 S4

# [5,5,12]->[5,5,120]
# 第三层 卷积层，这里需要进行卷积计算后的大小为[5,5,12]，其后的池化层将特征进行再一次压缩
filter3 = weight_variable([5, 5, 24, 120])
bias3 = bias_variable([120])
conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding='SAME')  # 5*5
h_conv3 = tf.nn.relu(conv3 + bias3)

# 后面2个全连接层，全连接层的作用在整个卷积神经网络中起到“分类器”的作用
# 即将学到的“分布式特征表示”映射 到样本标记空间的作用

# 权值参数
W_fc1 = weight_variable([5 * 5 * 120, 60])
# 偏置值
b_fc1 = bias_variable([60])
# 将卷积的输出展开
h_pool2_flat = tf.reshape(h_conv3, [-1, 5 * 5 * 120])
# 神经网络计算，并添加sigmoid激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 输出层，使用softmax进行多分类
# 这里对池化后的数据进行重新展开，将二维数据重新展开成一维数组之后计算每一行的元素个数。最后一个输出层在使用了softmax进行概率的计算
W_fc2 = weight_variable([60, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 最后是交叉熵作为损失函数，使用梯度下降来对模型进行训练
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# sess = tf.InteractiveSession()

# 测试正确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 求出最大值的坐标并做对比
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train capacity=10000 smile=5616 not_smile=4384
image_batch, label_batch = tt.get_tfrecord(200, '..//data/firstGround//train//smile_train.tfrecords', capacity=10000)
# train capacity=10000 smile=5616 not_smile=4384
image_total, label_total = tt.get_tfrecord(10000, '..//data/firstGround//train//smile_train.tfrecords', capacity=10000)
# test capacity=2995 smile=1632 not_smile=1363
image_batch_test, label_batch_test = tt.get_tfrecord(2995, '..//data/firstGround//test//smile_test.tfrecords',
                                                     capacity=2995)
sava_molel = './model/first_ground_model.ckpt'


def draw_leaning_line(iteration_time, training_cost, testing_cost):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(iteration_time, training_cost, 'r', label='training_cost')
    ax.plot(iteration_time, testing_cost, label='testing_cost')
    ax.legend(loc=1)
    ax.set_xlabel('iteration_time')
    ax.set_ylabel('cost')
    ax.set_title('training_cost & testing_cost')
    plt.show()


def draw_leaning_line2(iteration_time, train_accuracy, test_accuracy):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(iteration_time, train_accuracy, 'r', label='train_accuracy')
    ax.plot(iteration_time, test_accuracy, label='test_accuracy')
    ax.legend(loc=1)
    ax.set_xlabel('iteration_time')
    ax.set_ylabel('accuracy')
    ax.set_title('train_accuracy & test_accuracy')
    plt.show()


def trian():
    training_cost, testing_cost, iteration_time, train_accuracy, test_accuracy = [], [], [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 进行训练
        start_time = time.time()
        b_image_test, b_label_test = sess.run([image_batch_test, label_batch_test])
        b_image_total, b_label_total = sess.run([image_total, label_total])
        for i in range(1000):
            # 取训练数据
            b_image, b_label = sess.run([image_batch, label_batch])
            # 训练数据
            train_step.run(feed_dict={x_image: b_image, y_: b_label})
            # 每迭代10个batch,对当前训练数据进行测试，输出当前预测准确率
            if i % 4 == 0:
                trian_c = cross_entropy.eval(feed_dict={x_image: b_image_total, y_: b_label_total}) / 10000.0
                # print("step %d, training trian_cost %g" % (i, trian_c))
                test_c = cross_entropy.eval(feed_dict={x_image: b_image_test, y_: b_label_test}) / 2995.0
                # print("total test cost %g" % test_c)
                train_a = accuracy.eval(feed_dict={x_image: b_image_total, y_: b_label_total})
                # print("step %d, training accuracy %g" % (i, train_accuracy))
                test_a = accuracy.eval(feed_dict={x_image: b_image_test, y_: b_label_test})
                print("step %d,total test accuracy %g" % (i, test_a))
                training_cost.append(trian_c)
                testing_cost.append(test_c)
                train_accuracy.append(train_a)
                test_accuracy.append(test_a)
                iteration_time.append(i)
        draw_leaning_line(iteration_time, training_cost, testing_cost)
        draw_leaning_line2(iteration_time, train_accuracy, test_accuracy)
        end_time = time.time()
        print("total test accuracy %g" % test_a)
        print('time:', (end_time - start_time) / 60.0)
        saver = tf.train.Saver()
        saver.save(sess, sava_molel)
        coord.request_stop()
        coord.join(threads)


trian()
