from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import struct
import os
import pandas as pd
import numpy as np
import time

# line 12 ~ 30 :: ubyte to csv
if not os.path.isfile("./MNIST.csv"):
    fp_image = open('train-images.idx3-ubyte','rb')
    fp_label = open('train-labels.idx1-ubyte','rb')
    csv_file = open("MNIST.csv",'w',encoding="utf-8")

    mag, lbl_count = struct.unpack(">II", fp_label.read(8))  # unsigned int *2로
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
X = np_MNIST[:, 1:]
Y = np_MNIST[:, 0]
# print(np.shape(X))  # -> 59999

# scaling
X_255 = X/255
X_MinMax = MinMaxScaler().fit(X).transform(X)

# parmeter setting
debug_Mode = False
cv_ = 5

# model setting
linear_model = LinearSVC(max_iter=10000, verbose=debug_Mode)
non_linear_model = SVC(max_iter=10000, verbose=debug_Mode)

# Run
print("-----Cross validate :: cv is {} -----".format(cv_))

start1 = time.perf_counter()
linear_score = cross_val_score(linear_model, X_255, Y, cv=cv_)
end1 = time.perf_counter()
print("linear :: Acurracy is {}, {:.2f}sec".format(np.round(linear_score.mean(), 3), end1-start1))

start2 = time.perf_counter()
non_linear_score = cross_val_score(non_linear_model, X_255, Y, cv=cv_)
end2 = time.perf_counter()
print("non_linear :: Acurracy is {}, {:.2f}sec\n".format(np.round(non_linear_score.mean(), 3), end2-start2))

print("-----Cross validate with Min_Max_scaling:: cv is {} -----".format(cv_))

start3 = time.perf_counter()
linear_score_with_min_max = cross_val_score(linear_model, X_MinMax, Y, cv=cv_)
end3 = time.perf_counter()
print("linear :: Acurracy is {}, {:.2f}sec".format(np.round(linear_score_with_min_max.mean(), 3), end3-start3))

start4 = time.perf_counter()
non_linear_score_with_min_max = cross_val_score(non_linear_model, X_MinMax, Y, cv=cv_)
end4 = time.perf_counter()
print("non_linear :: Acurracy is {}, {:.2f}sec".format(np.round(non_linear_score_with_min_max.mean(), 3), end4-start4))
