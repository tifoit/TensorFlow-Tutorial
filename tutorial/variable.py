# coding=utf-8
'''
Created on 2016年11月22日
@author: joe
'''
import numpy as np
import tensorflow as tf


def method_one():
    state = tf.Variable(0, name='counter')  
    state2 = tf.Variable(0, name='counter2')
    # print (state.name)  
    one = tf.constant(1)  
  
    new_value = tf.add(state, one)  
    update = tf.assign(state, new_value)
    
    new_value_2 = tf.add(new_value, one) 
    new_value_3 = tf.add(new_value_2, one) 
    new_value_4 = tf.add(new_value_3, one) 
    update_2 = tf.assign(state2, new_value_4)

    init = tf.initialize_all_variables()  
    with tf.Session() as sess:  
        sess.run(init)  
        for _ in range(3):
            print ("update2=%s," % sess.run(update_2)),
            print ("state2=%s," % sess.run(state2)),
            print ("new_value_4=%s," % sess.run(new_value_4)),
            print ("update=%s" % sess.run(update))       

def method_two():
    # 创建一个常量op， 产生一个1x2矩阵，这个op被作为一个节点
    # 构造器的返回值代表该常量op的返回值
    matrix1 = tf.constant([[3., 3.]])
    # 创建另一个常量op, 产生一个2x1的矩阵
    matrix2 = tf.constant([[2.], [2.]])
    # 创建一个矩阵乘法matmul op，把matrix1和matrix2作为输入：
    product = tf.matmul(matrix1, matrix2)
    # 启动默认图
    sess = tf.Session()
    # 调用sess的'run()' 方法来执行矩阵乘法op，传入'product'作为该方法的参数
    # 上面提到，'product'代表了矩阵乘法op的输出，传入它是向方法表明，我们希望取回
    # 矩阵乘法op的输出。
    #
    # 整个执行过程是自动化的，会话负责传递op所需的全部输入。op通常是并发执行的。
    #
    # 函数调用'run(product)' 触发了图中三个op（两个常量op和一个矩阵乘法op）的执行。
    # 返回值'result'是一个numpy 'ndarray'对象。
    result = sess.run(product)
    print result
    # ==>[[12.]]

    # 完成任务，关闭会话
    sess.close()
    
if __name__ == '__main__':
    method_one()
