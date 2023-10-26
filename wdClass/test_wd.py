import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

# 输入一个表示文本文件路径的字符串，该函数用于将文本数据转换为一个NumPy数组
# 外层表示文本文件的行，内层表示每行的数据/字符
# 最后将列表转换为NumPy数组，将字符类型转换为整数类型，返回该数组
def txt2mat(txt_path):
    f = open(txt_path, encoding='utf-8')
    lines = [list(line.strip()) for line in f.readlines()]
    return np.array(lines).astype(int)

# 输入一个表示csv文件路径的字符串，该函数用于解析CSV文件，提取其中的文本ID和标签信息
# 返回两个列表，包含了CSV文件中的文本ID和标签信息
def parse_traincsv(csv_path):
    f = open(csv_path, encoding='utf-8')

    # 从csv文件的第一行（标题行）读取头文件并去除首尾的空白字符
    head = f.readline().strip()

    # 创建两个空列表来保存文本id和标签
    txtids = []
    labels = []

    # 读取所有行并随机打乱
    lines = f.readlines()
    random.shuffle(lines)

    # 遍历每一行，每行按逗号分隔，提取文本ID和标签，添加到两个列表中
    for line in lines:
        (txtid, label) = line.strip().split(',')
        txtids.append("{}.txt".format(txtid))
        labels.append(int(label))
    return txtids, labels

# 输入包含文件名的列表，可选的目录路径，默认为None
# 将文本文件列表中每个文件转换为NumPy数组，并将所有这些数组存储在一个大的Numpy数组中
# file_list是个包含文件名的列表
# 最后返回mat，是个包含了所有文本文件的NumPy数组
def get_all_txt(file_list, parentdir=None):

    # 创建一个全0数组mat，维度表示要处理的文本文件数量，每个文件的行数，每行的字符数。
    mat = np.zeros(shape=(len(file_list), 32, 32))
    for (idx, file) in enumerate(file_list):
        if parentdir: mat[idx, ...] = txt2mat(os.path.join(parentdir, file))
        else: mat[idx, ...] = txt2mat(file)
    return mat

# 输入dir是个表示目录路径的字符串
# 返回两个列表，包含所有文件名和包含去掉扩展名的文件名
def get_all_name(dir):
    txtfiles = os.listdir(dir)
    txtnames = [file.split('.')[0] for file in txtfiles]
    return txtfiles, txtnames

# 输入mat是一个NumPy数组，表示特征矩阵，label是个NumPy数组，表示标签，K是knn算法中的近邻数
# 用于训练KNN分类器，返回训练后的模型
def wdClassTrain(mat, label, k):
    print('KNN start training.')

    # 自动选择算法
    knn = KNN(n_neighbors=k, algorithm="auto")

    # 训练KNN模型
    knn.fit(mat, label)
    print('KNN end train.')
    score = knn.score(mat, label)
    print('KNN model score:', score)
    return knn





if __name__ == "__main__":
    print("current workspace dir : ", os.getcwd())
    traindir = "./wdClass/train"

    # 从csv文件中解析训练数据，包含文件名和标签
    files, labels = parse_traincsv("./wdClass/train.csv")

    # mat是个包含了所有数据的numpy矩阵
    mat = get_all_txt(files, parentdir=traindir)

    # 将原始图像数据重新塑造成一个二维矩阵，每行代表一个图像样本，每列代表像素值
    # 32*32是图像的尺寸，它将图像展平为一个长度为1024的向量
    rmat = mat.reshape(-1, 32*32)
    labels = np.array(labels)
    # print(rmat, labels)

    # 划分训练集和验证集，通常80用来训练，20用来验证机器学习模型
    split_point = int(rmat.shape[0]*0.8)
    (trainmat, trainlabels), (valmat, vallabels) = (rmat[:split_point],labels[:split_point]),(rmat[split_point:],labels[split_point:])


    # 使用训练集的数据和标签来训练KNN分类器，后续用于图像分类任务
    knn = wdClassTrain(trainmat, trainlabels, 11)

    # random choose a item to val
    r = random.choice(list(range(0, valmat.shape[0])))
    pred = knn.predict(valmat[r].reshape(-1, 1024))
    img = Image.fromarray((valmat[r]*255).reshape(32, 32))
    print(pred)

    # 显示图像
    plt.imshow(img)
    plt.show()

    # 在全部验证集上的分数
    valpreds = knn.predict(valmat)
    print("f1 score:", f1_score(vallabels, valpreds, average='micro'))
    print("acc:", accuracy_score(vallabels, valpreds))

    # 生成测试结果
    testdir = "./wdClass/test_no_label"
    testfiles, testnames = get_all_name(testdir)
    test_mat = get_all_txt(testfiles, testdir)

    preds = knn.predict(test_mat.reshape(-1, 32*32))
    
    testnames = np.array(testnames).astype(int)
    submit = np.vstack([testnames, preds])
    submit = pd.DataFrame(submit.T, columns=["Test_txt_name", "Digit"])
    submit.to_csv("./wdClass/submit.csv", index=None)

























