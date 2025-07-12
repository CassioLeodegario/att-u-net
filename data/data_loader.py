# data/data_loader.py

import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
import config # Importa o config para ter acesso às dimensões

def load_dataset(path):
    """
    Carrega os caminhos das imagens de treino e validação das pastas pré-divididas.
    """
    X_train = sorted(glob(os.path.join(path, "train", "*.png")))
    y_train = sorted(glob(os.path.join(path, "train_mask", "*.png")))

    X_val = sorted(glob(os.path.join(path, "val", "*.png")))
    y_val = sorted(glob(os.path.join(path, "val_mask", "*.png")))

    return (X_train, y_train), (X_val, y_val)

def read_image(path):
    """ Lê uma imagem do disco e a normaliza. """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    """ Lê uma máscara do disco. """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT))
    # A normalização já é feita implicitamente no notebook original, aqui mantemos
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    return x

def tf_parse(x, y):
    """ Função wrapper para o TensorFlow. """
    def _parse(x, y):
        x = read_image(x.numpy())
        y = read_mask(y.numpy())
        return x, y

    x, y = tf.py_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([config.HEIGHT, config.WIDTH, 3])
    y.set_shape([config.HEIGHT, config.WIDTH, 1])
    return x, y

def tf_dataset(X, Y, batch_size):
    """ Cria um objeto tf.data.Dataset. """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset