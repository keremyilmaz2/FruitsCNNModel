"""
Yardımcı Fonksiyonlar
Veri işleme, görselleştirme ve genel yardımcı fonksiyonlar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import cv2
import json
from datetime import datetime
import shutil
from pathlib import Path

def setup_gpu():
    """GPU ayarlarını yapılandır"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU kullanılıyor: {len(gpus)} adet GPU bulundu")
        except RuntimeError as e:
            print(f"GPU ayarlama hatası: {e}")
    else:
        print("GPU bulunamadı, CPU kullanılacak")

def get_class_names(data_path):
    """Sınıf isimlerini klasör isimlerinden al"""
    class_names = []
    if os.path.exists(data_path):
        class_names = sorted([d for d in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, d))])
    print(f"Toplam {len(class_names)} sınıf bulundu")
    return class_names

def count_images_per_class(data_path):
    """Her sınıfta kaç görüntü olduğunu say"""
    class_counts = {}
    class_names = get_class_names(data_path)
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(image_files)
    
    return class_counts

def plot_data_distribution(class_counts, save_path=None):
    """Veri dağılımını görselleştir"""
    plt.figure(figsize=(15, 8))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(range(len(classes)), counts)
    plt.title('Sınıf Başına Görüntü Sayısı Dağılımı')
    plt.xlabel('Sınıflar')
    plt.ylabel('Görüntü Sayısı')
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dağılım grafiği kaydedildi: {save_path}")
    
    plt.show()

def display_sample_images(data_path, class_names, samples_per_class=3):
    """Her sınıftan örnek görüntüler göster"""
    fig, axes = plt.subplots(len(class_names[:10]), samples_per_class, 
                            figsize=(15, 30))
    
    for i, class_name in enumerate(class_names[:10]):  # İlk 10 sınıf
        class_path = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for j in range(min(samples_per_class, len(images))):
            img_path = os.path.join(class_path, images[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{class_name}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_validation_split(train_path, val_path, split_ratio=0.2):
    """Training setinden validation seti oluştur"""
    if os.path.exists(val_path):
        print("Validation klasörü zaten mevcut!")
        return
    
    os.makedirs(val_path, exist_ok=True)
    class_names = get_class_names(train_path)
    
    for class_name in class_names:
        source_class_path = os.path.join(train_path, class_name)
        target_class_path = os.path.join(val_path, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # Sınıf içindeki tüm görüntüleri al
        images = [f for f in os.listdir(source_class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Validation için rastgele seç
        np.random.shuffle(images)
        val_count = int(len(images) * split_ratio)
        val_images = images[:val_count]
        
        # Validation klasörüne kopyala
        for img_name in val_images:
            source = os.path.join(source_class_path, img_name)
            target = os.path.join(target_class_path, img_name)
            shutil.copy2(source, target)
        
        print(f"{class_name}: {len(val_images)} görüntü validation'a kopyalandı")

def plot_training_history(history, save_path=None):
    """Eğitim geçmişini görselleştir"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy grafiği
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss grafiği
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eğitim grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Confusion matrix görselleştir"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.show()

def save_model_info(model, history, config, save_dir):
    """Model bilgilerini kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model summary'yi kaydet
    with open(os.path.join(save_dir, f'model_summary_{timestamp}.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Eğitim geçmişini kaydet
    history_dict = {key: [float(val) for val in values] 
                   for key, values in history.history.items()}
    
    with open(os.path.join(save_dir, f'training_history_{timestamp}.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Konfigürasyonu kaydet
    with open(os.path.join(save_dir, f'config_{timestamp}.json'), 'w') as f:
        # NumPy array'leri listeye çevir
        config_serializable = {}
        for key, value in config.items():
            if isinstance(value, dict):
                config_serializable[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                          for k, v in value.items()}
            else:
                config_serializable[key] = value if not isinstance(value, np.ndarray) else value.tolist()
        
        json.dump(config_serializable, f, indent=2)
    
    print(f"Model bilgileri kaydedildi: {save_dir}")

def preprocess_single_image(img_path, target_size=(100, 100)):
    """Tek bir görüntüyü önişle"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_single_image(model, img_path, class_names, top_k=5):
    """Tek bir görüntü için tahmin yap"""
    img = preprocess_single_image(img_path)
    predictions = model.predict(img)[0]
    
    # En yüksek k tahmini al
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = [(class_names[i], predictions[i]) for i in top_indices]
    
    return top_predictions

def calculate_model_size(model_path):
    """Model dosya boyutunu hesapla"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    return "Dosya bulunamadı"

def print_classification_report(y_true, y_pred, class_names, save_path=None):
    """Detaylı sınıflandırma raporu"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Sınıflandırma raporu kaydedildi: {save_path}")
    
    return report

if __name__ == "__main__":
    print("Utils modülü yüklendi!")
    print("GPU durumu kontrol ediliyor...")
    setup_gpu()