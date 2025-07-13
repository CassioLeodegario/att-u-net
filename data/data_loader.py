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

    # Ignora valores > 7 (transforma em 0)
    x = np.where(x > 7, 0, x)

    x = x.astype(np.int32)
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

def tf_augment(x, y):
    x = tf.image.random_flip_left_right(x)
    y = tf.image.random_flip_left_right(tf.expand_dims(y, -1))
    y = tf.squeeze(y, -1)

    x = tf.image.random_brightness(x, max_delta=0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x, y

def tf_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset