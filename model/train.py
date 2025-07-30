import sys
import os
att_unet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, att_unet_path)

import config
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from utils import metrics
import matplotlib.pyplot as plt

def train_model(model, train_dataset, valid_dataset, loss_function):
    metrics_to_use = [metrics.dice_coef, metrics.iou, "accuracy"]

    callbacks = [
        ModelCheckpoint(config.MODEL_PATH, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7),
        CSVLogger(config.RESULTS_PATH + "training_log.csv"),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]

    model.compile(
        loss=loss_function,
        optimizer=tf.keras.optimizers.Adam(config.LR),
        metrics=metrics_to_use
    )

    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    
    return history

def plot_history(history, results_path):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'loss_plot.png'))
    
    plt.figure()
    plt.plot(history.history['dice_coef'], label='Training Dice Coef')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coef')
    plt.legend()
    plt.savefig(os.path.join(results_path, "plot.png"))