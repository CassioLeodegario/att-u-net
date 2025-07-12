# main.py

import os
# IMPORTANTE: Se você quer usar a GPU no Vast.ai, REMOVA ou comente a linha abaixo.
# Se você setar para -1, ele rodará na CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from data import data_loader
from model import unet, train # Assumindo que 'train' é seu módulo com a função train_model e plot_history
import config

if __name__ == "__main__":
    # Garante que o backend do Keras está sendo usado (já é padrão para TF 2.x)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Oculta mensagens de log do TF menos importantes

    # Cria a pasta de resultados se não existir
    # Certifique-se que config.RESULTS_PATH está definido no seu config.py
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
        print(f"Diretório de resultados criado: {config.RESULTS_PATH}")
    
    # Verifica e configura a disponibilidade de GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configura o crescimento de memória para evitar que a GPU aloque toda a memória de uma vez
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        except RuntimeError as e:
            # Erros de memória devem ser capturados aqui
            print(f"Erro ao configurar GPU: {e}")
    else:
        print("Nenhuma GPU encontrada. O treinamento será executado na CPU.")


    # Carregando o dataset e realizando a divisão para treino, validação e teste
    # Use config.DATASET_BASE_PATH que aponta para a pasta raiz do seu dataset (ex: /workspace/dataset/BCSS_512)
    # A variável test_split_ratio é passada para o data_loader.load_dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        data_loader.load_dataset(config.DATASET_PATH, test_split_ratio=0.15)
    
    # Ajuste o print para as variáveis corretas
    print(f"Dataset carregado:")
    print(f"  Treino: {len(X_train)} imagens")
    print(f"  Validação: {len(X_val)} imagens")
    print(f"  Teste: {len(X_test)} imagens")


    # Criando os datasets do TensorFlow
    # Use X_train, y_train, X_val, y_val conforme retornados por data_loader.load_dataset
    train_dataset = data_loader.tf_dataset(X_train, y_train, config.BATCH_SIZE)
    valid_dataset = data_loader.tf_dataset(X_val, y_val, config.BATCH_SIZE)
    
    # Opcional: Se você quiser um dataset para inferência/previsão, use tf_dataset_inference
    # test_inference_dataset = data_loader.tf_dataset_inference(X_test, config.BATCH_SIZE)


    # Construindo o modelo
    input_shape = (config.HEIGHT, config.WIDTH, 3) # As dimensões da imagem e canais
    # Certifique-se que unet.build_unet existe e retorna um modelo Keras
    model = unet.build_unet(input_shape, config.NUM_CLASSES)
    model.summary()

    # Treinando o modelo
    # Certifique-se que train.train_model existe e espera o modelo e os datasets
    history = train.train_model(model, train_dataset, valid_dataset)

    # Salvando os gráficos do histórico de treinamento
    # Certifique-se que train.plot_history existe e espera o objeto history
    train.plot_history(history, config.RESULTS_PATH) # Passando RESULTS_PATH para salvar no local correto
    
    # Salvar o modelo após o treinamento
    # Certifique-se que config.MODEL_PATH está definido no seu config.py
    model.save(config.MODEL_PATH)
    
    print("\nTreinamento concluído!")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Log de treinamento e gráficos salvos em: {config.RESULTS_PATH}")