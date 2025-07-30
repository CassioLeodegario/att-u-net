import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

import config  # Garante que NUM_CLASSES (que agora é 5) seja importado

# --- Métricas de Avaliação Modificadas ---
# Estas métricas agora calculam o score apenas para as 4 classes de interesse,
# ignorando a 5ª classe ("ignorar").

def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Calcula o coeficiente Dice médio para as classes de interesse.
    """
    NUM_INTEREST_CLASSES = 4  # As 4 classes que queremos avaliar

    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    
    # Corta os tensores para incluir apenas as classes de interesse (0 a 3)
    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]
    
    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return K.mean(dice)

def iou(y_true, y_pred, smooth=1e-7):
    """
    Calcula o IoU (Intersection over Union) médio para as classes de interesse.
    """
    NUM_INTEREST_CLASSES = 4

    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
    
    # Corta os tensores para incluir apenas as classes de interesse
    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]

    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2]) - intersection
    iou_score = (intersection + smooth) / (union + smooth)

    return K.mean(iou_score)

# --- Funções de Perda ---
# A dice_loss também precisa ser ajustada para corresponder à métrica
# e garantir que a perda de Dice também foque nas classes corretas.

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Calcula a perda de Dice focada nas classes de interesse.
    """
    NUM_INTEREST_CLASSES = 4
    
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)

    # Corta os tensores para incluir apenas as classes de interesse
    y_true_interest = y_true[..., :NUM_INTEREST_CLASSES]
    y_pred_interest = y_pred[..., :NUM_INTEREST_CLASSES]

    intersection = K.sum(y_true_interest * y_pred_interest, axis=[1, 2])
    union = K.sum(y_true_interest + y_pred_interest, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - K.mean(dice)

@register_keras_serializable()
def focal_loss(gamma=2.0):
    """
    A Focal Loss não precisa ser modificada aqui, pois a ponderação
    que aplicamos em main.py (com peso zero para a classe a ignorar)
    já cuida de focar na perda das classes corretas.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn

def combined_loss(y_true, y_pred):
    """
    Esta função agora usará a dice_loss modificada.
    A parte da focal loss será gerenciada pela ponderação de pesos em main.py.
    """
    # Esta função está definida em main.py, mas se você a usasse daqui,
    # ela se beneficiaria automaticamente da dice_loss modificada.
    # f_loss = focal_loss(gamma=2.0)(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return d_loss # Exemplo: se quisesse apenas a dice loss