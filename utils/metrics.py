import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

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

@register_keras_serializable()
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        # One-hot em y_true
        y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)

        # Estabilização numérica
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # Focal Loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # média por imagem
    return loss_fn

def combined_loss(y_true, y_pred):
    """
    Combinação de Focal Loss e Dice Loss.
    """
    f_loss = focal_loss(gamma=2.0)(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return f_loss + d_loss