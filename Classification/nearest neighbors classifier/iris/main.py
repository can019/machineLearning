import numpy as np

iris_data_set = np.loadtxt('iris.csv', delimiter=",")

# print(iris_data_set)

# Calculate Uclidean distance
# meanVector is mean of iris_train_set vector (without label)
"""
def cal_distance_with_meanVector(data, group_num):
    result = np.sqrt(np.sum(np.square(mean_vector[group_num] - data[0:-1])))
    return result
"""
def knn_classify(train_set, test_set, k):
    return 0


max_iteration_count = 5
test_data_size = np.size(iris_data_set, 0)/max_iteration_count

print(type(max_iteration_count))
# K-Cross Validation
for iteration_count in range(max_iteration_count):
    # slice
    first_index_of_test_set = int(iteration_count*test_data_size)
    last_index_of_test_set = int(test_data_size+iteration_count*test_data_size)

    test_data_set = \
        iris_data_set[first_index_of_test_set:last_index_of_test_set]
    train_data_set = \
        np.concatenate((iris_data_set[0:first_index_of_test_set], iris_data_set[last_index_of_test_set:-1]), axis=0)

    print(test_data_set)
    print("---------------------")
    print(train_data_set)
    print("---------------------")
    print(first_index_of_test_set,last_index_of_test_set)
    print("---------------------")
