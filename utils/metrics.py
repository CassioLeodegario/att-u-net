# utils/metrics.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return K.mean(iou)