# importing libraries
import sys
import numpy as np
from tabulate import tabulate

import warnings

warnings.filterwarnings("ignore")

# loading dataset
filename = sys.argv[1]
with open(filename, "r") as f:
    dataset = np.load("data.npy", allow_pickle=True)

# splitting the dataset for training and testing
np.random.shuffle(dataset)
train_data, test_data = np.split(dataset, [1200])


# testing knn using custom class

from knn import KNN

k = 7
distance_metric = "cosine"
encoder_type = "resnet"

knnobj = KNN(k, distance_metric, encoder_type, train_data, test_data)
accuracy, precision, recall, f1 = knnobj.calc_scores(
    knnobj.predict(displayPred=False, metrics=False), display=False
)

# printing metrics in tabular form
table_data = [
    ["Metric", "Score"],
    ["Accuracy", "{:.2f}%".format(accuracy * 100)],
    ["Precision", "{:.2f}%".format(precision * 100)],
    ["Recall", "{:.2f}%".format(recall * 100)],
    ["F1 Score", "{:.2f}%".format(f1 * 100)],
]
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
