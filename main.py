# main.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from data import data_loader
from model import unet, train
import config

if __name__ == "__main__":
    # Garante que o backend do Keras está sendo usado
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Cria a pasta de resultados se não existir
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
        
    # Verifica a disponibilidade de GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Atualmente, não é necessário definir a memória virtual, mas é uma boa prática
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        except RuntimeError as e:
            print(e)

    # Carregando o dataset das pastas pré-divididas
    (train_x, train_y), (valid_x, valid_y) = data_loader.load_dataset(config.DATASET_PATH)
    print(f"Dataset carregado: {len(train_x)} treino, {len(valid_x)} validação.")

    # Criando os datasets do TensorFlow
    train_dataset = data_loader.tf_dataset(train_x, train_y, config.BATCH_SIZE)
    valid_dataset = data_loader.tf_dataset(valid_x, valid_y, config.BATCH_SIZE)

    # Construindo o modelo
    input_shape = (config.HEIGHT, config.WIDTH, 3)
    model = unet.build_unet(input_shape)
    model.summary()

    # Treinando o modelo
    history = train.train_model(model, train_dataset, valid_dataset)

    # Salvando os gráficos do histórico de treinamento
    train.plot_history(history)
    
    print("\nTreinamento concluído!")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Log de treinamento e gráficos salvos em: {config.RESULTS_PATH}")