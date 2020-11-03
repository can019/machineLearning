import numpy as np
import struct
import os
import csv
import gzip
import time
#  전역변수

#  gz 파일 압축 해제
def unzip(src, dst):
    if not os.path.isfile(src):
        print("fatal :: {}가 존재하지 않습니다.".format(src))
        return 0
    print("  -----------------------------------  ")
    print("  압축 해제 시작 :: src = {}".format(dst))
    input = gzip.GzipFile(src, 'rb')
    s = input.read()
    input.close()
    output = open(dst, 'wb')
    output.write(s)
    output.close()
    print("  압축 해제 성공 :: dst = {}".format(dst))
    print("  -----------------------------------  ")


def pre_processing():
    print("[[-----------------------------------]]")
    print("전처리 시작")
    start_pre_processing = time.perf_counter()
    src1 = './train-images-idx3-ubyte.gz'
    dst1 = './train-images.idx3-ubyte'

    src2 = './train-labels-idx1-ubyte.gz'
    dst2 = './train-labels.idx1-ubyte'

    #  line 17, 20파일에 대한 압축해제
    if not os.path.isfile(dst1):  # 존재하지 않으면 압축해제
        unzip(src1, dst1)
    if not os.path.isfile(dst2):  # 존재하지 않으면 압축해제
        unzip(src2, dst2)

    if not os.path.isfile("./MNIST.csv"):
        print("  -----------------------------------")
        print("  MNIST.csv 파일 존재하지 않음. 변환시작")
        try:
            train_file = open('train-images.idx3-ubyte', 'rb')
            label_file = open('train-labels.idx1-ubyte', 'rb')
            csv_file = open("MNIST.csv", 'w', encoding="utf-8")

            train_file_size = os.path.getsize("train-labels.idx1-ubyte")  # label 수와 x 데이터 수가 같기에 한번만 size check
            writer = csv.writer(csv_file)
            # writer.writerow(np.zeros(785).tolist())  # pandas를 위한 더미 행 추가 numpy로 읽을 것이기 때문에 이번엔 제거
            x_cursor, y_cursor = train_file.read(16), label_file.read(8)  # cusor 옮겨놓기
            for i in range((train_file_size - 8)):
                x_cursor = train_file.read(784)
                y_cursor = label_file.read(1)

                temp_x = list(struct.unpack(len(x_cursor) * 'B', x_cursor))  # byte->int, list형으로 저장
                temp_y = list(struct.unpack(len(y_cursor) * 'B', y_cursor))  # byte->int, list형으로 저장
                temp_y.extend(temp_x)  # 합치기
                writer.writerow(temp_y)  # csv에 작성
            csv_file.close()
            label_file.close()
            train_file.close()
            print("  변환 완료")
        except:
            print("  변환 실패")
    print("  -----------------------------------")
    print("  Mnist 읽기 시작")
    start_read_file = time.perf_counter()
    MNIST = np.loadtxt("MNIST.csv", delimiter=",")
    end_read_file = time.perf_counter()
    print("  Mnist 읽기 완료 :: {}sec".format(end_read_file-start_read_file))

    target_labels = np.array([1, 5, 8])

    # slice data :: X_set, label_set
    slice_by_1_5_8 = slice_by_target_label(MNIST, target_labels)

    # shuffle
    mixed_data = np.random.permutation(slice_by_1_5_8)

    # One-Hot Encoding
    after_one_hot_encoding = one_hot_encoding(mixed_data, target_labels)

    print("  -----------------------------------")
    print("  train set, test set 분리 시작")
    X = mixed_data[:, 1:]/255.
    total_data_size = np.size(X, axis=0)
    train_X = X[:int(total_data_size*0.6), :]
    test_X = X[int(total_data_size*0.6):, :]
    train_label = after_one_hot_encoding[:int(total_data_size*0.6), :]
    test_label = after_one_hot_encoding[int(total_data_size*0.6), :]
    print("  train set, test set 분리 끝")

    end_pre_processing = time.perf_counter()
    print("  -----------------------------------")
    print("전처리 끝 :: {}sec".format(end_pre_processing-start_pre_processing))
    print("[[-----------------------------------]]")

    return train_X, test_X, train_label, test_label

def one_hot_encoding(src, labels):
    print("  -----------------------------------")
    print("  One-Hot Encoding 시작")
    dst = np.zeros([np.size(src, axis=0), np.size(labels)])
    for i in range(np.size(src, axis=0)):
        target = src[i][0]
        if target == labels[0]:
            dst[i][0] = 1
        elif target == labels[1]:
            dst[i][1] = 1
        else:
            dst[i][2] = 1
    print("  One-Hot Encoding 끝")
    return dst

def slice_by_target_label(src, target):
    print("  -----------------------------------")
    print("  특정 label에 따른 dataset 채택 시작")
    dst = src[src[:, 0] == target[0]]
    for i in range(1, np.size(target)):
        temp = src[src[:, 0] == target[i]]
        dst = np.concatenate([dst, temp])
    print("특정 label에 따른 dataset 채택 완료 :: {}".format(np.shape(dst)))
    return dst

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initialize(feature_size, node_num, output_size):
    weight1 = np.random.normal(size=node_num * (feature_size + 1))
    weight1 = weight1.reshape(node_num, feature_size + 1)
    weight2 = np.random.normal(size=output_size * (node_num + 1))
    weight2 = weight2.reshape(output_size, node_num + 1)
    return weight1, weight2

def stochastic(train_set, label_set, epoch, learning_rate, hidden_layer_size):
    # TODO :: 가중치 초기화
    input = np.empty([785])
    weight1,  weight2 = initialize(784, hidden_layer_size, np.size(label_set, axis=1))
    for sequence in range(epoch):  # epoch = 5~10
        print("-------------epoch :: {}---------------".format(sequence))  # epoch 출력
        
        # TODO :: data 섞기
        train_set, label_set = shuffle_data(train_set, label_set)
        print("섞기 완료")
        input[1:] = train_set[0]
        input[0] = 1
        right_answer = label_set[0]
        print("임의 데이터 추출 완료")

        # TODO :: 순전파
        zsum = np.dot(weight1, input)  # 전방 1 (은닉)
        layer1_output = sigmoid(zsum)
        hidden_layer_nodes = np.ones(hidden_layer_size+1)
        hidden_layer_nodes[1:] = layer1_output
        print("은닉층 준비 완료")
        osum = np.dot(weight2, hidden_layer_nodes)
        output_layer = sigmoid(osum)
        print("전방 완료")

        # TODO :: 역전파
        print("역전파 시작")
        delta_k = np.zeros(np.shape(osum))
        delta_u2 = np.zeros(np.shape(weight2))
        etha = np.zeros(hidden_layer_size)
        delta_u1 = np.zeros(np.shape(weight1))

        for k in range(3):
            delta_k[k] = (right_answer[k]-output_layer[k]) * sigmoid_grad(osum[k])

        for k in range(3):
            for j in range(hidden_layer_size+1):
                delta_u2[k, j] = -delta_k[k] * hidden_layer_nodes[j]

        for j in range(1, hidden_layer_size):
            delta_sigma_u2_sum = 0
            for q in range(3):
                delta_sigma_u2_sum = delta_sigma_u2_sum+delta_k[q]*delta_u2[q][j]
            etha[j] = delta_sigma_u2_sum * sigmoid_grad(zsum[j])

        # # hidden layer = p
        # # c = 3
        # # d = input
        for j in range(hidden_layer_size):
            for i in range(np.size(input)):
                delta_u1 = -etha[j]*input[i]
        print("역전파 종료")

        print("경사하강 시작")

        weight1 = weight1 - learning_rate*delta_u1
        weight2 = weight2 - learning_rate*delta_u2
        print("경사하강 끝")

    return weight1, weight2

def cal_accuracy(model, X_set, label_set): # 성능측정
    print()

def shuffle_data(x, label):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    label = label[s]
    return x, label


if __name__ == '__main__':
    # a = np.array([[1,2,0],[4,5,1],[1,3,2],[1,3,3],[1,3,4]])
    # s = np.arange(a.shape[0])
    # np.random.shuffle(s)
    # b = np.array([1,2][0,1,2,3,4])
    # a = a[s]

    train_X, test_X, train_label, test_label = pre_processing()  # 전처리
    stochastic(train_X, train_label, 10, 0.01, 50)
    # TODO epoch마다 가중치 print
    # TODO 학습 후 train_set에 대한 정확도 출력
    # TODO 학습 후 test_set에 대한 정확도 출력

