import numpy as np
from pathlib import Path
import os
import pandas as pd
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras


instance_num = 10

# cifar10
def get_cifar10_data(x_test):
    x_test = x_test.astype('float32') / 255.0
    w, h = 32, 32
    x_test = x_test.reshape(x_test.shape[0], w, h, 3)
    return x_test
_, (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = get_cifar10_data(x_test)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


dataset_dir = Path("") / 'cifar10'
dataset_dir.mkdir(exist_ok=True)
np.savez(str(dataset_dir / "inputs.npz"), x_test[:instance_num])
np.savez(str(dataset_dir / "ground_truths.npz"), y_test[:instance_num])
print('------------ CIFAR-10 ---------------')
print(x_test[:instance_num].shape)
print(y_test[:instance_num].shape)

# MNIST
def get_mnist_data(x_test):
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_test
_, (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = get_mnist_data(x_test)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


dataset_dir = Path("") / 'mnist'
dataset_dir.mkdir(exist_ok=True)
np.savez(str(dataset_dir / "inputs.npz"), x_test[:instance_num])
np.savez(str(dataset_dir / "ground_truths.npz"), y_test[:instance_num])
print('------------ MNIST ---------------')
print(x_test[:instance_num].shape)
print(y_test[:instance_num].shape)

# SineWave
dataframe = pd.read_csv("sinewave.csv")
test_size, seq_len = 1500, 50
data_test = dataframe.get("sinewave").values[-(test_size + 50):]
data_windows = []
for i in range(test_size):
    data_windows.append(data_test[i:i + seq_len])
data_windows = np.array(data_windows).astype(float).reshape((test_size, seq_len, 1))
data_windows = np.array(data_windows).astype(float)
x_test = data_windows[:, :-1]
y_test = data_windows[:, -1, [0]]

dataset_dir = Path("") / 'sinewave'
dataset_dir.mkdir(exist_ok=True)
np.savez(str(dataset_dir / "inputs.npz"), x_test[:instance_num])
np.savez(str(dataset_dir / "ground_truths.npz"), y_test[:instance_num])
print('------------ SineWave ---------------')
print(x_test[:instance_num].shape)
print(y_test[:instance_num].shape)


