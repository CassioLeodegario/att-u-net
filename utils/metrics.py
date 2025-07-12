import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K
import config  

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return K.mean(dice)

def iou(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return K.mean(iou)

def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - K.mean(dice)