# data_loader.py

import sys
import os
att_unet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, att_unet_path)

import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import config

def load_dataset(path, test_split_ratio=0.15):
    all_X_train_val = sorted(glob(os.path.join(path, "train_512", "*.png")))
    all_y_train_val = sorted(glob(os.path.join(path, "train_mask_512", "*.png")))

    X_val_from_folder = sorted(glob(os.path.join(path, "val_512", "*.png")))
    y_val_from_folder = sorted(glob(os.path.join(path, "val_mask_512", "*.png")))

    X_train, X_test, y_train, y_test = train_test_split(
        all_X_train_val, all_y_train_val,
        test_size=test_split_ratio,
        random_state=42
    )

    X_val = X_val_from_folder
    y_val = y_val_from_folder

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_NEAREST)

    # Reindexar os valores para [0, ..., 18]
    label_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13,
        15: 14, 17: 15, 18: 16, 19: 17, 20: 18
    }
    x = np.vectorize(label_map.get)(x).astype(np.int32)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x.numpy())
        y = read_mask(y.numpy())
        return x, y

    x, y = tf.py_function(_parse, [x, y], [tf.float32, tf.int32])
    x.set_shape([config.HEIGHT, config.WIDTH, 3])
    y.set_shape([config.HEIGHT, config.WIDTH])
    return x, y

def tf_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
