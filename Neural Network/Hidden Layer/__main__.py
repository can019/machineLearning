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

    target_labels = np.array([1,5,8])

    # slice data :: X_set, label_set
    slice_by_1_5_8 = slice_by_target_label(MNIST, target_labels)

    # shuffle
    mixed_data = np.random.permutation(slice_by_1_5_8)

    # One-Hot Encoding
    after_one_hot_encoding = one_hot_encoding(mixed_data, target_labels)

    print("  -----------------------------------")
    print("  train set, test set 분리 시작")
    X = mixed_data[:, 1:]
    total_data_size = np.size(X,axis=0)
    train_X = X[:int(total_data_size*0.6), :]
    test_X = X[int(total_data_size*0.6):, :]
    train_label = after_one_hot_encoding[:, :int(total_data_size*0.6), :]
    test_label = after_one_hot_encoding[:, int(total_data_size*0.6), :]
    print("  train set, test set 분리 끝")

    end_pre_processing = time.perf_counter()
    print("  -----------------------------------")
    print("전처리 끝 :: {}sec".format(end_pre_processing-start_pre_processing))
    print("[[-----------------------------------]]")

    return train_X, test_X, train_label, test_label

def one_hot_encoding(src, labels):
    print("  -----------------------------------")
    print("  One-Hot Encoding 시작")
    dst = np.empty([np.size(labels)-1, np.size(src, axis=0), np.size(labels)])
    for i in range(np.size(labels)-1):
        temp = np.zeros([np.size(src, axis=0), np.size(labels)])
        for j in range(np.size(src, axis=0)):
            if src[j][0] == labels[i]:
                temp[j][i] = 1
        dst[i] = temp
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

if __name__ == '__main__':
    # a = np.array([[1,2,3],[4,5,3],[1,3,7]])
    # np.random.shuffle(a)
    # print(a)
    # print(np.random.permutation(a))

    train_X, test_X, train_label, test_label = pre_processing()  # 전처리
    

