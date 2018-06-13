#!/usr/bin/python3
# -*- encoding: utf-8 -*-


import numpy as np


# 该数据集取自《统计学习方法》例8.1
DATA_FILE = "adaboost_data.txt"


def load_data(data_file=DATA_FILE):
    with open(data_file, 'r') as fin:
        lines = fin.readlines()
        data_list = []
        for line in lines:
            data_list.append(line.strip().split(' '))

    data_list = np.array(data_list, dtype="float")
    if fin is not None:
        fin.close()

    return data_list


def load_data_test():
    print(load_data())


# 两种基本的决策树桩
def classify_type1(data, threshold):
    if data <= threshold:
        return 1
    else:
        return -1


def classify_type2(data, threshold):
    if data <= threshold:
        return -1
    else:
        return 1


def adaboost(data, m):
    """
    param: data is the input data type of numpy.array
    param: m is the numbers of basic classifier
    return: basic classifier list, correspond classifier coefficients
    """
    data_len = len(data)
    # 初始化数据权重
    weight_list = []
    for _ in range(data_len):
        weight_list.append(1 / data_len)
    # 基本分类器的权重系数
    classifier_coeff = []
    """
    AdaBoost中一个基本分类器是一个二叉树桩，且最终的结果是所有基本分类器的加权和
    因此不使用树结构，而使用一个m长度的列表，记录每个分割点的值
    分割点的选取：在对x排序的情况下，以0.5为步长遍历x选取
    (该例子中，输入数据向量x仅有一维)
    由于有两种基本的决策树桩，因此在classifier中要指定使用哪一种
    """
    classifier = []
    x_list = data[:, 0]
    sorted_x_list = sorted(x_list)
    labels = np.array(data[:, -1], dtype="int")
    current_split_point = sorted_x_list[0]

    for i in range(m):
        # 计算分割点
        """
        注意！！！这里的基本分类器可使用 if(x < k) return 1 else return -1
        以及 if(x > k) return 1 else return -1两种，具体使用哪种取决于
        哪种的错误率在更低
        """
        # 采用哪种决策树桩
        classifier_type = -1
        min_error_rate = 1
        for x in sorted_x_list:
            error_rate_type1 = .0
            error_rate_type2 = .0
            index = 0
            # 计算错误率
            for item in x_list:
                # 分别计算两种基本分类器的分类错误率
                if classify_type1(item, x + 0.5) != labels[index]:
                    error_rate_type1 += weight_list[index]
                if classify_type2(item, x + 0.5) != labels[index]:
                    error_rate_type2 += weight_list[index]
                index += 1
            # 选取最佳分割点
            if error_rate_type1 < min_error_rate:
                min_error_rate = error_rate_type1
                classifier_type = 1
                current_split_point = x + 0.5
            if error_rate_type2 < min_error_rate:
                min_error_rate = error_rate_type2
                classifier_type = 2
                current_split_point = x + 0.5

        # 第i轮的基本分类器和分类器系数
        classifier.append({current_split_point: classifier_type})
        coeff = 1 / 2 * np.log((1 - min_error_rate) / min_error_rate)

        classifier_coeff.append(coeff)

        # 更新数据分布的权值

        # 规范化因子
        Z = 0
        if classifier_type == 1:
            for j in range(data_len):
                Z += weight_list[j] * np.exp(- coeff * labels[j]
                                             * classify_type1(x_list[j], current_split_point))
            for j in range(data_len):
                weight_list[j] = weight_list[j] * np.exp(- coeff * labels[j]
                                                         * classify_type1(x_list[j], current_split_point)) / Z
        elif classifier_type == 2:
            for j in range(data_len):
                Z += weight_list[j] * np.exp(- coeff * labels[j]
                                             * classify_type2(x_list[j], current_split_point))
            for j in range(data_len):
                weight_list[j] = weight_list[j] * np.exp(- coeff * labels[j]
                                                         * classify_type2(x_list[j], current_split_point)) / Z

        # 在测试集上能够正确分类的个数
        error_now = test(data, classifier, classifier_coeff)
        print("Current errors count on the training set is %d " % error_now)
        if error_now <= 0:
            return classifier, classifier_coeff

    return classifier, classifier_coeff


def test(data, classifier, classifier_coeff):
    # 分类错误个数
    count = 0
    x = np.array(data[:, 0])
    labels = np.array(data[:, -1], dtype="int")
    assert (len(classifier) == len(classifier_coeff)), "invalid parameters!"
    classifier_amount = len(classifier)
    size = len(data)
    for index in range(size):
        result = .0
        for i in range(classifier_amount):
            classifier_type = list(classifier[i].values())[0]
            split_point = list(classifier[i].keys())[0]
            if classifier_type == 1:
                result += (classifier_coeff[i] 
                    * classify_type1(x[index], split_point))
            else:
                result += (classifier_coeff[i] 
                * classify_type2(x[index], split_point))
        if int(np.sign(result)) != labels[index]:
            count += 1

    return count


if __name__ == "__main__":
    data = load_data()
    classifier, classifier_coeff = adaboost(load_data(), 20)
    print(classifier)
    print(classifier_coeff)
