import numpy as np
import struct
import os
import csv


if not os.path.isfile("./MNIST.csv"):

    train_file = open('train-images.idx3-ubyte', 'rb')
    label_file = open('train-labels.idx1-ubyte', 'rb')
    csv_file = open("MNIST.csv", 'w', encoding="utf-8")

    train_file_size = os.path.getsize("train-labels.idx1-ubyte")  # label 수와 x 데이터 수가 같기에 한번만 size check
    writer = csv.writer(csv_file)
    writer.writerow(np.zeros(785).tolist())  # pandas를 위한 더미 행 추가
    x_cursor, y_cursor = train_file.read(16), label_file.read(8)  # cusor 옮겨놓기
    for i in range((train_file_size-8)):
        x_cursor = train_file.read(784)
        y_cursor = label_file.read(1)

        temp_x = list(struct.unpack(len(x_cursor) * 'B', x_cursor))  # byte->int, list형으로 저장
        temp_y = list(struct.unpack(len(y_cursor) * 'B', y_cursor))  # byte->int, list형으로 저장
        temp_y.extend(temp_x)  # 합치기
        writer.writerow(temp_y)  #csv에 작성

    csv_file.close()
    label_file.close()
    train_file.close()
