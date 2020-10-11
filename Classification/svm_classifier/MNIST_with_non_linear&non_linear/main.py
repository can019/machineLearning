from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import struct
import os
import numpy as np
import pandas as pd
# line 7 ~ 25 :: ubyte to csv
if not os.path.isfile("./MNIST.csv"):
    fp_image = open('train-images.idx3-ubyte','rb')
    fp_label = open('train-labels.idx1-ubyte','rb')
    csv_file = open("MNIST.csv",'w',encoding="utf-8")

    mag, lbl_count = struct.unpack(">II", fp_label.read(8))
    mag, img_count = struct.unpack(">II", fp_image.read(8))
    rows, cols = struct.unpack(">II", fp_image.read(8))
    pixels = rows*cols
    res = []
    for idx in range(lbl_count):
        label = struct.unpack("B", fp_label.read(1))[0]
        bdata = fp_image.read(pixels)
        sdata = list(map(lambda n: str(n), bdata))
        csv_file.write(str(label) + ",")
        csv_file.write(",".join(sdata) + "\r\n")
    csv_file.close()
    fp_label.close()
    fp_image.close()

# read data
# pandas로 읽은 후 numpy로 변환하는 것이 numpy로 바로 읽는 것보다 7초 이상 빨랐습니다.
read_MNIST = pd.read_csv("MNIST.csv")  # pandas로 읽음
np_MNIST = read_MNIST.to_numpy()  # numpy로 변환

# slice data
x_data = np_MNIST[:,1:]
y_data = np_MNIST[:,0]
print()
