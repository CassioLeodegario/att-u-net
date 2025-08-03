import os
import tensorflow as tf
from data import data_loader
from model import unet, train
import config
import numpy as np
from tqdm import tqdm
from utils import metrics

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
        print(f"Diretório de resultados criado: {config.RESULTS_PATH}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")
    else:
        print("Nenhuma GPU encontrada. O treinamento será executado na CPU.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        data_loader.load_dataset(config.DATASET_PATH, test_split_ratio=0.15)
    
    print(f"\nDataset carregado:")
    print(f"  Treino: {len(X_train)} imagens")
    print(f"  Validação: {len(X_test)} imagens")
    print(f"  Teste (hold-out): {len(y_val)} imagens") 

    print("\n[INFO] Analisando a distribuição de classes no conjunto de treino (após remapeamento)...")
    class_counts = {i: 0 for i in range(config.NUM_CLASSES)}
    
    for mask_path in tqdm(y_train, desc="Analisando máscaras"):
        mask = data_loader.read_mask(mask_path.encode())
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls in class_counts:
                class_counts[cls] += count

    print("\nDistribuição de pixels por classe (após remapeamento):")
    total_pixels = sum(class_counts.values())
    labels = ["tumor", "stroma", "lymphocytic_infiltrate", "necrosis_or_debris", "IGNORAR"]
    
    for cls, label in enumerate(labels):
        count = class_counts.get(cls, 0)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  Classe {cls} ({label}): {count} pixels ({percentage:.4f}%)")
    
    class_frequencies = np.array([class_counts.get(i, 0) for i in range(config.NUM_CLASSES)], dtype=np.float32)
    
    class_weights = np.zeros_like(class_frequencies)
    
    interest_frequencies = class_frequencies[:4]
    non_zero_frequencies = interest_frequencies[interest_frequencies > 0]
    
    if len(non_zero_frequencies) > 0:
        median_frequency = np.median(non_zero_frequencies)
        
        class_weights[:4] = median_frequency / (interest_frequencies + 1e-6) 

    print("\nPesos calculados para as classes (peso 0 para ignorar):")
    for i, label in enumerate(labels):
        print(f"  {label}: {class_weights[i]:.4f}")
    
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def weighted_combined_loss(y_true, y_pred):
        y_true_one_hot = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=config.NUM_CLASSES), tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred_clipped)
        weighted_cross_entropy = cross_entropy * class_weights_tensor
        focal_modulation = tf.pow(1 - y_pred_clipped, 2.0)
        f_loss = focal_modulation * weighted_cross_entropy
        f_loss_reduced = tf.reduce_mean(tf.reduce_sum(f_loss, axis=-1))

        d_loss = metrics.dice_loss(y_true, y_pred)
        
        return f_loss_reduced + d_loss

    train_dataset = data_loader.tf_dataset(X_train, y_train, config.BATCH_SIZE)
    valid_dataset = data_loader.tf_dataset(X_test, y_test, config.BATCH_SIZE)
    
    input_shape = (config.HEIGHT, config.WIDTH, 3)
    model = unet.build_unet(input_shape, config.NUM_CLASSES)
    model.summary()

    history = train.train_model(
        model, 
        train_dataset, 
        valid_dataset, 
        loss_function=weighted_combined_loss 
    )

    train.plot_history(history, config.RESULTS_PATH)
    
    model.save(config.MODEL_PATH)
    
    print("\nTreinamento concluído!")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Log de treinamento e gráficos salvos em: {config.RESULTS_PATH}")