# utils/metrics.py

import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    """
    Calcula o Coeficiente de Dice.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def iou(y_true, y_pred):
    """
    Calcula o IoU (Intersection over Union).
    """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(float)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)