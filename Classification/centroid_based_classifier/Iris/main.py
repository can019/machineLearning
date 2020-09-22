import numpy as np

iris_train_set = np.loadtxt('iris_train.csv', delimiter=",")
iris_test_set = np.loadtxt('iris_test.csv', delimiter=",")

# Calculate Uclidean distance
# meanVector is mean of iris_train_set vector (without label)
def cal_distance_with_meanVector(data, group_num):
    result = np.sqrt(np.sum(np.square(mean_vector[group_num] - data[0:-1])))
    return result


# how many items in class
count = np.zeros([3, 4])

# get sum of vectors.
sum_of_vector = np.zeros([3, 4])

# meanVector has three group's vector. ([class1], [class2], [class3])
# class shpae is [w, x, y, z]
mean_vector = np.zeros([3, 4])
# loop for get sun of vectors from iris_train_set
for i in range(np.size(iris_train_set, 0)):
    if iris_train_set[i][4] == 1.:
        sum_of_vector[0] = sum_of_vector[0] + iris_train_set[i][0:-1]
        count[0] = count[0]+1

    elif iris_train_set[i][4] == 2.:
        sum_of_vector[1] = sum_of_vector[1] + iris_train_set[i][0:-1]
        count[1] = count[1] + 1

    elif iris_train_set[i][4] == 3.:
        sum_of_vector[2] = sum_of_vector[2] + iris_train_set[i][0:-1]
        count[2] = count[2] + 1
# divde sum_of_vector by count to get meanVector
mean_vector = sum_of_vector/count

distance = np.zeros([3, 1])
correct_count = 0

# classify iris_test_set by mean vector
for i in range(np.size(iris_test_set, 0)):
    label = iris_test_set[i][-1]
    distance[0] = cal_distance_with_meanVector(iris_test_set[i], 0)
    distance[1] = cal_distance_with_meanVector(iris_test_set[i], 1)
    distance[2] = cal_distance_with_meanVector(iris_test_set[i], 2)

    if np.argmin(distance)+1 == label:
        correct_count = correct_count+1

# print Accuracy
print("----------------------------------")
print("정확도 : {:0.3f}%".format(correct_count/30*100, "."))