import random
from sklearn.datasets import load_digits
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
plt.imshow(random_digit_image, cmap='gray', interpolation='nearest')
plt.axis("off")
plt.show()

