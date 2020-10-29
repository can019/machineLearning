import numpy as np
import struct
import os
import csv
import gzip

#  gz 파일 압축 해제
def unzip(src, dst):
    if not os.path.isfile(src):
        print("fatal :: {}가 존재하지 않습니다.".format(src))
        return 0
    print("-----------------------------------")
    print("압축 해제 시작 :: src = {}".format(dst))
    input = gzip.GzipFile(src, 'rb')
    s = input.read()
    input.close()
    output = open(dst, 'wb')
    output.write(s)
    output.close()
    print("압축 해제 성공 :: dst = {}".format(dst))
    print("-----------------------------------")


def pre_processing():
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
        print("-----------------------------------")
        print("MNIST.csv 파일 존재하지 않음. 변환시작")
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
            print("변환 완료")
        except:
            print("변환 실패")
        finally:
            print("-----------------------------------")
        print("Mnist 읽기 시작")
        MNIST = np.loadtxt("MNIST.csv", delimiter=",")
        print("Mnist 읽기 완료")

        # slice data :: X_set, label_set
        X = MNIST[:, :-1]
        y = MNIST[:, -1]

if __name__ == '__main__':
    pre_processing()
