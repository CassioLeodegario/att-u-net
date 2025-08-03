import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

import config  

def dice_coef(y_true, y_pred, smooth=1e-7):
    NUM_INTEREST_CLASSES = 4  

    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    
    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]
    
    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return K.mean(dice)

def iou(y_true, y_pred, smooth=1e-7):
    NUM_INTEREST_CLASSES = 4

    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    
    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]

    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2]) - intersection
    iou_score = (intersection + smooth) / (union + smooth)

    return K.mean(iou_score)

def dice_loss(y_true, y_pred, smooth=1e-7):
    NUM_INTEREST_CLASSES = 4
    
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)

    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]

    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - K.mean(dice)

@register_keras_serializable()
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn