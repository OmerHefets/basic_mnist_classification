import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt

mnist_digits = load_digits()
X, y = mnist_digits["data"], mnist_digits["target"]
print(X.shape)
print(y.shape)

random_digit_index = random.randint(1, X.shape[0])
random_digit = X[random_digit_index]
random_digit_image = random_digit.reshape(8, 8)

print(y[random_digit_index])
# plt.imshow(random_digit_image, cmap='gray', interpolation='nearest')
# plt.axis("off")
# plt.show()

# 80% training, 20% test
X_train, X_test, y_train, y_test = X[:1440], X[1440:], y[:1440], y[1440:]

# shuffle data
shuffled_indexes = np.random.RandomState(seed=42).permutation(1440)
X_train, y_train = X_train[shuffled_indexes], y_train[shuffled_indexes]

# training only for one digit
y_train_3 = (y_train == 3)
y_test_3 = (y_test == 3)

SGD_mnist = SGDClassifier(random_state=42)
SGD_mnist.fit(X_train, y_train_3)


y_train_3_estimates = cross_val_score(SGD_mnist, X_train, y_train_3, scoring="accuracy", cv=4)
y_train_3_predictions = cross_val_predict(SGD_mnist, X_train, y_train_3, cv=3)

print(y_train_3_estimates.shape)
print(y_train_3_predictions.shape)

print("Decision: {}".format(precision_score(y_train_3, y_train_3_predictions)))
print("Recall: {}".format(recall_score(y_train_3, y_train_3_predictions)))


# practicing precision-recall
y_train_3_decision_scores = SGD_mnist.decision_function(X_train)
print(y_train_3_decision_scores)
print(y_train_3)


precision_y_train_3, recall_y_train_3, threshold = precision_recall_curve(y_train_3, y_train_3_decision_scores)


def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "r--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "b--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1.1])
    plt.xlim([-5000, 5000])
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def plot_roc_curve(fpr, tpr, fpr_forest, tpr_forest):
    plt.plot(fpr_forest, tpr_forest, "b-", label="Forest")
    plt.plot(fpr, tpr, "g-", label="SGD")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

# plot_precision_recall_threshold(precision_y_train_3, recall_y_train_3, threshold)
# plot_precision_vs_recall(precision_y_train_3, recall_y_train_3)

# ROC


FPR_sgd, TPR_sgd, _ = roc_curve(y_train_3, y_train_3_decision_scores)
# plot_roc_curve(FPR, TPR)

# trying the same with "random forests" classifier
Forest_mnist = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(Forest_mnist, X_train, y_train_3, cv=3, method="predict_proba")
y_probas_forest_score = y_probas_forest[:, 1]

FPR_forest, TPR_forest, thre = roc_curve(y_train_3, y_probas_forest_score)
plot_roc_curve(FPR_sgd, TPR_sgd, FPR_forest, TPR_forest)

# Multiclass classification
SGD_mnist.fit(X_train, y_train)
random_digit = X_train[127]
print(SGD_mnist.predict([random_digit]))
print(SGD_mnist.decision_function([random_digit]))
