import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images.shape
# print(len(train_labels))

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 这一步是创建神经网络，第一层是扁平化28*28的像素为一个128维的一维数组，
# 第二层Dense是稠密层，叫做稠密连接层或者全连接层，包含128个节点，也是128个神经元；
# 最后一层返回10个输出，有10个节点，代表每个概率。

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

# 数据被轮多少次就是多少epochs （从头开始轮的那种）

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test accuracy:", test_acc)

prediction = model.predict(test_images)

print(prediction[0])

print("------------------------------------------")
print(np.argmax(prediction[0]))
# 函数取到最大值时x的值的点集 argmax
