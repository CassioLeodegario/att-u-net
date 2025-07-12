# model/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from . import config
from utils import metrics
import matplotlib.pyplot as plt

def train_model(model, train_dataset, valid_dataset):
    """
    Compila e treina o modelo com os datasets fornecidos.
    """
    # Métricas
    metrics_to_use = [metrics.dice_coef, metrics.iou, "accuracy"]

    # Callbacks
    callbacks = [
        ModelCheckpoint(config.MODEL_PATH, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger(config.RESULTS_PATH + "training_log.csv"),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]

    # Compilação
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(config.LR),
        metrics=metrics_to_use
    )

    # Treinamento
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    
    return history

def plot_history(history):
    """
    Salva gráficos da perda e do coeficiente de Dice ao longo das épocas.
    """
    # Gráfico da Perda (Loss)
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(config.RESULTS_PATH + 'loss_plot.png')
    
    # Gráfico do Coeficiente de Dice
    plt.figure()
    plt.plot(history.history['dice_coef'], label='Training Dice Coef')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coef')
    plt.legend()
    plt.savefig(config.RESULTS_PATH + 'dice_coef_plot.png')