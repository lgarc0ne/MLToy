#!/usr/bin/python3
# -*- encoding: utf-8 -*-


import numpy as np


DATA_FILE = "breast_cancer.txt"
FEATURES_FILE = "breast_cancer_features.txt"


def load_data(data_file_name=DATA_FILE):
    result = []
    with open(data_file_name, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            result.append(line.strip().split(','))
        fin.close()

    result = np.array(result)

    return result


def load_features(feature_file_name=FEATURES_FILE):
    result = []
    with open(feature_file_name, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            result.append(line.strip().split(','))
        fin.close()
    
    result = np.array(result)

    return result


def entropy(label_list):
    result_dict = {}
    size = len(label_list)
    for item in label_list:
        if item not in result_dict.keys():
            result_dict[item] = 1
        else:
            result_dict[item] += 1

    result = 0
    for key in result_dict.keys():
        result += - result_dict[key] / size * np.log2(result_dict[key] / size)

    return result


def calc_info_gain(data_array, ratio=False):
    """
    计算信息增益
    ratio为True时计算信息增益率
    @input:包含labels在内的整个数据列表
    @output:一个长度等于属性个数的list
    """
    label_list = data_array[:, -1]
    ent = entropy(label_list)
    result_list = []
    feature_size = len(data_array[0]) - 1
    size = len(data_array)
    for i in range(feature_size):
        current_feature_list = data_array[:, i]
        current_feature_dict = {}
        index = 0
        for item in current_feature_list:
            # 这里使用一个类似二维数组的，‘二维’字典
            if item not in current_feature_dict.keys():
                current_feature_dict[item] = {label_list[index]: 1}
            else:
                if label_list[index] not in current_feature_dict[item].keys():
                    current_feature_dict[item][label_list[index]] = 1
                else:
                    current_feature_dict[item][label_list[index]] += 1
            index += 1
        lack_amount = 0
        if '?' in current_feature_dict.keys():
            for label in current_feature_dict['?'].keys():
                lack_amount += current_feature_dict['?'][label]
            # 删除缺失数据节点
            current_feature_dict.pop('?')
        new_size = size - lack_amount
        cond_entropy = 0.0
        # 计算信息增益率

        emprical_feature_entropy = 0.0
        for key in current_feature_dict.keys():
            feature_entropy = 0.0
            # 该属性所占总数量
            # 注意：python3的dict.values()返回dict_values类型，并非一个list
            # 而numpy的这些函数都是以list为参数，故需要强行转换为list
            feature_count = np.sum(list(current_feature_dict[key].values()))
            if ratio:
                emprical_feature_entropy += - feature_count / new_size * np.log2(feature_count / new_size)
            for label in current_feature_dict[key].keys():
                feature_entropy += - current_feature_dict[key][label] / feature_count * np.log2(current_feature_dict[key][label] / feature_count)
            # 按照缺失值的处理方法，乘上非缺失值占总数的比例    
            cond_entropy += feature_count / new_size * feature_entropy * (new_size / size)
        if ratio:
            # 信息增益率 
            emprical_feature_entropy *= (new_size / size)
            result_list.append((ent - cond_entropy) / emprical_feature_entropy)
        else:
            result_list.append(ent - cond_entropy)
    
    return result_list


def calc_info_gain_ratio(data_array):
    """
    计算信息增益率
    """
    return calc_info_gain(data_array, ratio=True)
        
    
def load_data_test():
    data = load_data()
    print(entropy(data[:, -1]))
    cond_ent = calc_info_gain(data)
    print(cond_ent)
    print(np.argmax(cond_ent))
    cond_ent_ratio = calc_info_gain_ratio(data)
    print(cond_ent_ratio)
    print(np.argmax(cond_ent_ratio))


if __name__ == '__main__':
    load_data_test()