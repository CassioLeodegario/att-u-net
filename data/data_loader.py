# data/data_loader.py

import sys
import os
att_unet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, att_unet_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split # Esta importação AGORA SERÁ USADA!
import config

def load_dataset(path, test_split_ratio=0.15): # Adiciona um argumento para a proporção do teste
    """
    Carrega os caminhos das imagens e máscaras, e divide os dados de treino
    para criar um conjunto de teste.
    """
    # Carrega todas as imagens e máscaras de treino e validação
    # Supondo que 'train_512' e 'train_mask_512' contêm a MAIORIA dos seus dados
    all_X_train_val = sorted(glob(os.path.join(path, "train_512", "*.png")))
    all_y_train_val = sorted(glob(os.path.join(path, "train_mask_512", "*.png")))

    # Carrega também o que está nas pastas de validação (se existirem e forem separadas)
    # É importante garantir que não há sobreposição entre estas e as "train_512"
    X_val_from_folder = sorted(glob(os.path.join(path, "val_512", "*.png")))
    y_val_from_folder = sorted(glob(os.path.join(path, "val_mask_512", "*.png")))

    # Combina todos os dados de treino/validação para fazer a divisão
    # Se você já tem uma divisão explícita de treino/validação em pastas,
    # considere se faz sentido combiná-los aqui ou apenas trabalhar com 'all_X_train_val'
    # para a divisão de teste. Para este exemplo, vamos considerar 'all_X_train_val' como o pool principal.
    
    # Divide o conjunto de treino original em um novo conjunto de treino e um conjunto de teste
    # Stratify é importante para garantir que a proporção de classes seja mantida (se aplicável)
    # random_state garante que a divisão seja a mesma cada vez que você rodar
    X_train, X_test, y_train, y_test = train_test_split(
        all_X_train_val, all_y_train_val,
        test_size=test_split_ratio,
        random_state=42 # Para reprodutibilidade
        # stratify=all_y_train_val # Use se sua task tiver classes desbalanceadas e 'all_y_train_val' for um array de labels
    )
    
    # Se você já tinha um conjunto de validação em pastas separadas, você usa ele.
    # Caso contrário, se 'all_X_train_val' era TODO o seu dataset, você precisaria
    # fazer outra divisão para validação.
    # Para simplicidade, vamos considerar que X_val_from_folder e y_val_from_folder são seus dados de validação reais.
    X_val = X_val_from_folder
    y_val = y_val_from_folder
    
    # Se você NÃO TEM pastas de validação separadas e quer dividir de 'all_X_train_val'
    # Esta seria uma abordagem alternativa:
    # X_temp, X_test, y_temp, y_test = train_test_split(all_X_train_val, all_y_train_val, test_size=test_split_ratio, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42) # Ex: 15% da parte restante para validação

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def read_image(path):
    """ Lê uma imagem do disco e a normaliza. """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    """ Lê uma máscara do disco. Para segmentação multiclasse. """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_NEAREST)
    x = x.astype(np.int32)  
    return x

def tf_parse(x, y):
    """ Função wrapper para o TensorFlow. """
    def _parse(x, y):
        x = read_image(x.numpy())
        y = read_mask(y.numpy())
        return x, y

    x, y = tf.py_function(_parse, [x, y], [tf.float32, tf.int32])
    x.set_shape([config.HEIGHT, config.WIDTH, 3])
    y.set_shape([config.HEIGHT, config.WIDTH]) 
    return x, y

def tf_dataset(X, Y, batch_size):
    """ Cria um objeto tf.data.Dataset. """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset