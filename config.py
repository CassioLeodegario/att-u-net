# config.py

# Parâmetros de imagem
HEIGHT = 256
WIDTH = 256

# Hiperparâmetros do modelo
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4 # Learning Rate

# Caminhos de dados e resultados
DATASET_PATH = '/workspace/dataset/BCSS_512'
MODEL_PATH = "results/attention_unet_bcss.keras" # Caminho para salvar o modelo treinado
RESULTS_PATH = "results/" # Pasta para salvar gráficos e outros resultados

NUM_CLASSES = 22