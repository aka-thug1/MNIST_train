import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.keras import layers, Model, utils

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# 构建模型
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))(x)
model = Model(inputs, outputs)

# 自定义DataGenerator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, num_same=2, num_diff=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_same = num_same
        self.num_diff = num_diff
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x_batch = np.zeros((self.batch_size * (self.num_same + self.num_diff), *self.x.shape[1:]))
        y_batch = np.zeros(self.batch_size * (self.num_same + self.num_diff), dtype=int)
        for i, idx in enumerate(batch_indexes):
            x_1 = self.x[idx]
            label_1 = self.y[idx].argmax()
            for j in range(self.num_same):
                same_idx = np.random.choice(np.where(self.y.argmax(1) == label_1)[0])
                x_2 = self.x[same_idx]
                x_batch[i * (self.num_same + self.num_diff) + j * 2] = x_1
                x_batch[i * (self.num_same + self.num_diff) + j * 2 + 1] = x_2
                y_batch[i * (self.num_same + self.num_diff) + j * 2] = 1
            for j in range(self.num_diff):
                diff_idx = np.random.choice(np.where(self.y.argmax(1) != label_1)[0])
                x_2 = self.x[diff_idx]
                x_batch[i * (self.num_same + self.num_diff) + self.num_same * 2 + j] = x_2
                y_batch[i * (self.num_same + self.num_diff) + self.num_same + j] = 0
        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# 自定义对比损失函数
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = keras.backend.square(y_pred)
    margin_square = keras.backend.square(keras.backend.maximum(margin - y_pred, 0))
    return keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)

# 对比学习训练
batch_size = 128
epochs = 10
num_same = 2
num_diff = 1
generator = DataGenerator(x_train, y_train, batch_size, num_same=num_same, num_diff=num_diff)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=contrastive_loss)
model.fit(generator, epochs=epochs, verbose=2)

# 特征可视化
features = model.predict(x_test)
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)
plt.figure(figsize=(10, 10))
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_test.argmax(1))
plt.show()

# 构建MLP分类器
inputs = layers.Input(shape=(128,))
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(x)
classifier = Model(inputs, outputs)

# 训练MLP分类器
classifier.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
classifier.fit(features, y_test, batch_size=batch_size, epochs=epochs, verbose=2)

# 输出测试集上的准确率
accuracy = classifier.evaluate(features, y_test, verbose=0)[1]
print(f"Accuracy on test set: {accuracy}")
