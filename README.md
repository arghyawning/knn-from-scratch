# K-Nearest Neighbours

### Task 1

Draw a graph that shows the distribution of the various labels across the entire dataset.

![graph obtained](label_freq_graph.png)

### Task 2

Create a KNN class where you implement the following: You should not use sklearn for this.

1. Create a class where you can modify and access the encoder type, k, and distance metric (and any required parameter) of the class
2. Return the inference (prediction) when given the above parameters (encoder type, k, and distance metric).
3. Return the validation f-1 score, accuracy, precision, and recall after splitting the provided dataset into train and val subsets. You are allowed to use sklearn metrics for this part.

Examples:

- KNN(encoder_type='resnet', k=3, distance_metric='euclidean')
  Accuracy: 21.00%
  Precision: 14.19%
  Recall: 14.16%
  F1 Score: 12.94%
- KNN(encoder_type='vit', k=5, distance_metric='manhattan')
Accuracy: 26.33%
Precision: 17.84%
Recall: 18.31%
F1 Score: 16.80%
<!-- - KNN(encoder_type='resnet', k=7, distance_metric='cosine') -->

### Task 3

Hyperparameter Tuning

1. Find the best (k, encoder, distance metric) triplet that gives the best validation accuracy for a given data split (your choice).
2. Print an Ordered rank list of top 20 such triplets.
3. Plot k vs accuracy given a choice(yours) of any given distance, encoder pair (with a constant data split).
