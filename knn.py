import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class KNN:
    def __init__(self, k, distance_metric, encoder_type, train_data, test_data):
        self.k = k  # number of nearest neighbors
        self.distance_metric = (
            distance_metric  # distance metric to use: euclidean, manhattan, cosine
        )
        self.encoder_type = encoder_type  # encoder type to use: resnet, vit
        self.train_data = train_data  # training data
        self.test_data = test_data
        self.labels_train = train_data[:, 3]
        self.labels_test = test_data[:, 3]
        self.test_values = (
            test_data[:, 1] if encoder_type == "resnet" else test_data[:, 2]
        )
        self.train_values = (
            train_data[:, 1] if encoder_type == "resnet" else train_data[:, 2]
        )

    # calculate the distance between the training and testing datapoint
    def calc_distance(self, testdp, traindp):
        if self.distance_metric == "euclidean":
            sub = testdp - traindp
            subsq = np.square(sub)
            subsqsum = np.sum(subsq)
            dist = np.sqrt(subsqsum)
            return dist
        elif self.distance_metric == "manhattan":
            sub = testdp - traindp
            subabs = np.abs(sub)
            dist = np.sum(subabs)
            return dist
        elif self.distance_metric == "cosine":
            dot = np.dot(testdp, traindp)
            testdpnorm = np.linalg.norm(testdp)
            traindpnorm = np.linalg.norm(traindp)
            dist = dot / (testdpnorm * traindpnorm)
            return dist
        else:
            print("Invalid distance metric")
            return

    # calculate accuracy, precision, recall and f1
    def calc_scores(self, pred_labels):
        accuracy = accuracy_score(self.labels_test, pred_labels)
        precision = precision_score(self.labels_test, pred_labels, average="macro")
        recall = recall_score(self.labels_test, pred_labels, average="macro")
        f1 = f1_score(self.labels_test, pred_labels, average="macro")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    # displays and returns predicted labels
    def predict(self, metrics):
        pred_labels = []
        for dptest in self.test_values:
            distances = {}
            index = 0
            for dptrain in self.train_values:
                dist = self.calc_distance(dptest, dptrain)
                distances[index] = dist
                index += 1
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])

            k_nearest_labels = {}
            maxfreq = 0
            maxfreqlabels = []
            for i in range(self.k):
                curr_label = self.labels_train[sorted_distances[i][0]]
                k_nearest_labels[curr_label] = (
                    k_nearest_labels.get(self.labels_train[sorted_distances[i][0]], 0)
                    + 1
                )
            for l, c in k_nearest_labels.items():
                if c > maxfreq:
                    maxfreq = c
                    pred_label = l
            pred_labels.append(pred_label)

        print("Predicted labels:")
        print(pred_labels)
        print("Actual labels:")
        print(self.labels_test.tolist())

        if metrics:
            self.calc_scores(pred_labels)

        return pred_labels
