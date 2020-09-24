import numpy as np

iris_data_set = np.loadtxt('iris.csv', delimiter=",")
iris_data_set = np.random.permutation(iris_data_set) # shuffle
# print(iris_data_set)

# Calculate Uclidean distance

def cal_distance(test_point, train_set):
    distance_numpy = np.sqrt(np.sum(np.square(train_set[:, 0:-1]-test_point[0:-1]), axis=1))
    return distance_numpy

def knn_classify(train_set, test_set, k):
    classified = np.array([], np.int32)
    for i in range(np.size(test_data_set, 0)):
        distance = cal_distance(test_set[i], train_set)
        k_nearest_nieghbors = np.array([], np.int32)

        for j in range(k):
            target_index = np.argmin(distance)
            k_nearest_nieghbors = np.append(k_nearest_nieghbors, int(train_set[target_index][-1]))
            distance = np.delete(distance, target_index)
        bin_count = np.bincount(k_nearest_nieghbors)
        classified = np.append(classified, np.argmax(bin_count))

    return classified


max_iteration_count = 5
test_data_size = np.size(iris_data_set, 0)/max_iteration_count

# K-Cross Validation
for iteration_count in range(max_iteration_count):
    correct_count = 0
    # slice
    first_index_of_test_set = int(iteration_count*test_data_size)
    last_index_of_test_set = int(test_data_size+iteration_count*test_data_size)

    test_data_set = \
        iris_data_set[first_index_of_test_set:last_index_of_test_set]
    train_data_set = \
        np.concatenate((iris_data_set[0:first_index_of_test_set], iris_data_set[last_index_of_test_set:]), axis=0)

    result = knn_classify(train_data_set, test_data_set, 1)

    for i in range(int(test_data_size)):
        if result[i] == int(test_data_set[i][-1]):
            correct_count = correct_count+1

    print("{}번째 :  {}%".format(iteration_count, round(correct_count/test_data_size*100,3)))

