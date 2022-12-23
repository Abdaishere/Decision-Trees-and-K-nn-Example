import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")


def euclidean_distance(row1, row2):
    distance = np.sqrt(np.sum((row1 - row2) ** 2))
    return distance


def get_neighbors(train, test_row, Y_train, k_neighbors):
    distances = []  # calculate distances from a test sample to every sample in a training set
    for i in range(len(train)):
        distances.append((Y_train[i], euclidean_distance(train[i], test_row)))
    distances.sort(key=lambda x: x[1])  # sort in ascending order, based on a distance value
    neighbors = []
    # 0 0 1 0 1
    for i in range(k_neighbors):  # get first k samples
        neighbors.append(distances[i][0])
    return neighbors


def predict(train, test_set, Y_train, num_neighbors):
    predictions = []
    for test_sample in test_set:
        labels = get_neighbors(train, test_sample, Y_train, num_neighbors)
        # labels = [sample for sample in neighbors]
        # 0 3  1 3
        prediction = max(labels, key=labels.count)
        predictions.append(prediction)
    return predictions


def calc_accuracy(actual, predicted):
    return np.sum(actual == predicted) / len(predicted)


dataset = pd.read_csv("BankNote_Authentication.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#  f(v) = (v - mean) / std
x_normalized = ((x - np.mean(x, axis=0)) / np.std(x, axis=0))

# Split the normalized data
X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, y, test_size=30 / 100)

for i in range(9):
    prediction = predict(X_train, X_test, Y_train, i + 1)
    acc = calc_accuracy(Y_test, prediction)
    print("Iteration With K = ", i + 1)
    print("Number of correctly classified instances : ", acc * len(prediction), " Total number of instances : ",
          len(prediction))
    print("Our Model Accuracy = ", acc * 100)

    knn = KNeighborsClassifier(n_neighbors=i + 1)
    # Fit the model
    knn.fit(X_train, Y_train)
    knn_acc = knn.score(X_test, Y_test) * 100
    print("Sklearn Model Accuracy = ", knn_acc)
    print("============================================================")
