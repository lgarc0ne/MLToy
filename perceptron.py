#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
感知机学习过程中的动态更新示意图
其中最小二乘直线的截距b未能精确计算出来
由于数据非线性可分，因此学习算法不可能收敛
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_data(data_file_name="testSet.txt"):
    with open(data_file_name, 'r') as fin:
        if fin is None:
            return list
        lines = fin.readlines()
        ret_matrix = np.zeros((len(lines), 3))
        index = 0
        for line in lines:
            ret_matrix[index, :] = line[:-1].split('\t')
            # 实际上原数据里负例值为0，会导致梯度无法更新
            # 故先将其改为-1
            if ret_matrix[index, -1] == 0:
                ret_matrix[index, -1] = -1
            index += 1
        
    return ret_matrix


def load_data_test():
    data = load_data()
    draw_plot(data)


def draw_plot(data_narray):
    figure, ax = plt.subplots()
    plt.title("Perceptron learning progress")
    plt.xlabel("X")
    plt.ylabel("Y")
    x = np.arange(-5, 4, 1)
    # 取正例和负例
    true_items = np.array([item for item in data_narray if item[2] == 1.0])
    false_items = np.array([item for item in data_narray if item[2] != 1.0])
    
    ax.scatter(true_items[:, 0], true_items[:, 1], marker='*')
    ax.scatter(false_items[:, 0], false_items[:, 1], marker='x')
    line = None
    # 三个参数的初始值
    para_dict = [1.0, 1.0, 1.0]
    line, = ax.plot([] , [], 'r-', linewidth=1)
    # 去掉lable
    input_mat = np.matrix(data_narray[:, 0:2])

    lable = np.matrix(data_narray[:,-1])
    # 用公式直接求最小二乘参数
    result_para = np.dot(np.dot(np.dot(input_mat.T, input_mat).I, input_mat.T), lable.T)
    print(result_para)
    line2, = ax.plot([],[], 'g')


    def update(i):
        label = 'timestamp {0}'.format(i)

        for item in data_narray:
            w1 = para_dict[0]
            w2 = para_dict[1]
            b = para_dict[2]
           
            if (w1 * item[0] + w2 * item[1] + b) * item[2] <= 0:
                # 对于不能正确判别的例子，使用随机梯度下降，更新其参数
                para_dict[0] = w1 + 0.2 * item[2] * item[0]
                para_dict[1] = w2 + 0.2 * item[2] * item[1]
                para_dict[2] =  b + 0.2 * item[2]
                w1 = para_dict[0]
                w2 = para_dict[1]
                b = para_dict[2]
                print(w1, w2)
                line.set_data(x, ((w1 * x) + b) / -w2)
                # 由于求b会导致矩阵奇异，大概赋值了一个
                line2.set_data(x, (result_para[0] * x + 0.39) / - result_para[1])
                ax.set_xlabel(label)
                return line, ax

   
    anim = FuncAnimation(
                figure, update, frames=np.arange(0, 1000), interval=1)
    plt.show()


if __name__ == "__main__":
    load_data_test()
