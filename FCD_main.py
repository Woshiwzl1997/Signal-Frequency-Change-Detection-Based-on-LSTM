# -*- coding: utf-8 -*-
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import h5py
hidden1_size=10 #lstm中隐藏节点的个数
hidden2_size=20

dnn_size=10

timesteps=64
training_steps=1300#训练轮数
batch_size=32# batch大小

testing_examples=186#测试数据个数

trainpath="traindata.mat"
testpath="testdata.mat"

def read_datat(trainpath,testpath):
    traindata=h5py.File(trainpath)
    testdata=h5py.File(testpath)

    train_sig=np.array(traindata["train_sig"],dtype=np.float32).T
    train_sig=np.expand_dims(train_sig,2)
    train_lab = np.array(traindata["train_lab"], dtype=np.float32).T

    test_sig=np.array(testdata["test_sig"],dtype=np.float32).T
    test_sig=np.expand_dims(test_sig,2)
    test_lab = np.array(testdata["test_lab"], dtype=np.float32).T

    return train_sig,train_lab,test_sig,test_lab

def fully_connected(prev_layer,num_units):
    layer=tf.layers.dense(prev_layer,num_units,use_bias=True,activation=tf.nn.tanh)
    output=tf.layers.dense(layer,1,activation=None)
    return output

def lstm_model(X,y,is_training):

    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden1_size),tf.nn.rnn_cell.BasicLSTMCell(hidden2_size)])
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

    predictions=fully_connected(output,dnn_size)
    if not is_training:
        return predictions,None,None
    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op=tf.train.RMSPropOptimizer(0.001).minimize(loss)
    return predictions,loss,train_op

def train(sess,train_X,train_y):

    ds=tf.data.Dataset.from_tensor_slices((train_X,train_y))
    ds=ds.repeat().shuffle(1000).batch(batch_size)#repeat:遍历数据集次数 shuffle:混乱程度
    X,y=ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果
    with tf.variable_scope("model"):
        predictions,loss,train_op=lstm_model(X,y,True)

    #初始化变量
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(training_steps):
        _,l=sess.run([train_op,loss])#使用dataset而不是placeholder导入数据
        if i%100==0:
            print("train step:"+str(i)+",loss"+str(l))
    #保存模型
    saver.save(sess, "model\model.ckpt")

def run_eval(sess,test_X,test_y):
    ds=tf.data.Dataset.from_tensor_slices((test_X,test_y))
    ds=ds.batch(1)
    X,y=ds.make_one_shot_iterator().get_next()

    # 加载图

    with tf.variable_scope("model"):
        prediction,_,_=lstm_model(X,[0.0],False)

        saver = tf.train.Saver()
        saver.restore(sess, "model\model.ckpt")

        predictions=[]
        labels=[]
        for i in range(testing_examples):
            p,l=sess.run([prediction,y])
            predictions.append(p)
            labels.append(l)
        predictions=np.array(predictions).squeeze()#2维变一维
        labels=np.array(labels).squeeze()
        rmse=np.sqrt(((predictions-labels)**2).mean(axis=0))
        print("MSE is :%f" % rmse)

        #画图
        plt.figure()
        plt.xlabel('时间')
        plt.ylabel('跳变点概率')
        plt.plot(predictions,label='predictions')
        plt.plot(labels,label='real')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    t_lab=False
    train_X, train_y, test_X, test_y = read_datat(trainpath, testpath)

    with tf.Session() as sess:
        #训练模型
        if t_lab:
            train(sess,train_X,train_y)
        #进行预测
        else:
            run_eval(sess,test_X,test_y)