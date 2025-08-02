

"""
Fruits-360 CNN Model Konfigürasyonu - TRANSFER RESNET50 OPTIMAL
En yüksek doğruluk için Transfer Learning kullanarak ayarlanmış
"""
import os

# Veri seti yolları (sizin mevcut yollarınız)
DATA_CONFIG = {
    'dataset_path': './FruitsData/fruits-360_100x100',
    'train_path': './FruitsData/fruits-360_100x100/fruits-360/Training',
    'test_path': './FruitsData/fruits-360_100x100/fruits-360/Test',
    'validation_split': 0.2,  # Training'den % kaçını validation yapalım
}
# Görüntü parametreleri
IMAGE_CONFIG = {
    'image_size': (100, 100),
    'input_shape': (100, 100, 3),
    'color_channels': 3,
    'normalization': True,  # Transfer learning için kritik
}

# # Model parametreleri - TRANSFER RESNET50
# MODEL_CONFIG = {
#     'num_classes': 206,
#     'model_name': 'fruits_resnet50_optimal',
#     'architecture': 'transfer_resnet50',  # 🏆 EN İYİ MODEL
#     'dropout_rate': 0.3,  # Overfitting koruması
#     'batch_normalization': True,
# }

# # Eğitim parametreleri - TRANSFER LEARNING İÇİN OPTİMİZE
# TRAINING_CONFIG = {
#     'batch_size': 16,  # Transfer learning için ideal
#     'epochs': 50,      # Yeterli ama fazla değil
#     'learning_rate': 0.0001,  # Transfer learning için DÜŞÜK LR
#     'optimizer': 'adam',
#     'loss_function': 'categorical_crossentropy',
#     'metrics': ['accuracy'],  # ← DÜZELTİLDİ
#     'validation_split': 0.15,
#     'shuffle': True,
#     'early_stopping_patience': 12,  # Sabırlı early stopping
#     'reduce_lr_patience': 6,
#     'save_best_only': True,
# }


# # Data Augmentation - AKILLI VE KONSERVATIF
# AUGMENTATION_CONFIG = {
#     'use_augmentation': True,
#     'rotation_range': 10,      # Hafif rotasyon
#     'width_shift_range': 0.1,  # Minimal shift
#     'height_shift_range': 0.1,
#     'shear_range': 0.05,       # Çok az shear
#     'zoom_range': 0.1,         # Hafif zoom
#     'horizontal_flip': True,   # Meyvelerde mantıklı
#     'vertical_flip': False,    # Meyvelerde mantıksız
#     'fill_mode': 'nearest',
# }

# Model parametreleri - FINE-TUNING İÇİN
MODEL_CONFIG = {
    'num_classes': 206,
    'model_name': 'fruits_resnet50_finetune',  # ⭐ YENİ İSİM
    'architecture': 'transfer_resnet50_finetune',  # ⭐ YENİ MİMARİ
    'dropout_rate': 0.4,  # Daha fazla dropout
    'batch_normalization': True,
}

# Eğitim parametreleri - FINE-TUNING İÇİN OPTİMİZE
TRAINING_CONFIG = {
    'batch_size': 16,          # ⭐ 16'dan 8'e (daha stabil)
    'epochs': 30,             # ⭐ 50'den 30'a (daha az)
    'learning_rate': 0.00001, # ⭐ ÇOK DÜŞÜK LR (önemli!)
    'optimizer': 'adam',
    'loss_function': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'validation_split': 0.15,
    'shuffle': True,
    'early_stopping_patience': 8,   # ⭐ Daha sabırsız
    'reduce_lr_patience': 4,        # ⭐ Daha hızlı LR azaltma
    'save_best_only': True,
}

# Data Augmentation - DAHA AGRESIF
AUGMENTATION_CONFIG = {
    'use_augmentation': True,
    'rotation_range': 20,        # ⭐ 10'dan 20'ye
    'width_shift_range': 0.15,   # ⭐ 0.1'den 0.15'e
    'height_shift_range': 0.15,
    'shear_range': 0.1,          # ⭐ 0.05'ten 0.1'e
    'zoom_range': 0.15,          # ⭐ 0.1'den 0.15'e
    'horizontal_flip': True,
    'vertical_flip': False,      # Meyvelerde mantıksız
    'fill_mode': 'nearest',
}




# Dosya yolları
PATHS = {
    'models_dir': 'models',
    'logs_dir': 'logs',
    'results_dir': 'results',
    'checkpoints_dir': 'checkpoints',
    'predictions_dir': 'predictions',
}

# Model kaydetme ayarları
# Model kaydetme ayarları
SAVE_CONFIG = {
    'save_model': True,
    'save_weights': True,
    'save_history': True,
    'model_format': 'h5',
    'checkpoint_monitor': 'val_accuracy',  # En yüksek validation accuracy
    'checkpoint_mode': 'max',
    'save_best_only': True,  # ← BU SATIRIP EKLEYİN
}

# Görselleştirme ayarları - DETAYLI ANALİZ
VISUALIZATION_CONFIG = {
    'plot_history': True,
    'plot_confusion_matrix': True,
    'save_plots': True,
    'plot_sample_predictions': True,
    'samples_to_plot': 25,  # Daha fazla örnek analizi
}

# GPU ayarları
GPU_CONFIG = {
    'use_gpu': True,
    'memory_growth': True,
    'mixed_precision': False,  # CPU için kapalı
}

# Sınıf isimleri
CLASS_NAMES = []  # Otomatik detect edilecek

# Debug ayarları - DETAYLI LOG
DEBUG_CONFIG = {
    'verbose': 1,
    'log_level': 'INFO',
    'save_debug_images': True,  # Debug için aktif
    'print_model_summary': True,
}

def create_directories():
    """Gerekli klasörleri oluştur"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    print("Gerekli klasörler oluşturuldu.")

def get_config():
    """Tüm konfigürasyonu döndür"""
    return {
        'data': DATA_CONFIG,
        'image': IMAGE_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'paths': PATHS,
        'save': SAVE_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'gpu': GPU_CONFIG,
        'class_names': CLASS_NAMES,
        'debug': DEBUG_CONFIG,
    }

if __name__ == "__main__":
    create_directories()
    config = get_config()
    print("🏆 TRANSFER RESNET50 OPTIMAL KONFİGÜRASYONU YÜKLENDİ!")
    print("=" * 55)
    print(f"🏗️ Mimari: {config['model']['architecture']}")
    print(f"📊 Batch size: {config['training']['batch_size']}")
    print(f"🎓 Epochs: {config['training']['epochs']}")
    print(f"🧠 Learning rate: {config['training']['learning_rate']}")
    print(f"🔄 Dropout rate: {config['model']['dropout_rate']}")
    print(f"📈 Beklenen doğruluk: %93-97")
    print(f"⏱️ Tahmini süre: 1.5-2 saat")
    print("=" * 55)
    print("Bu ayarlar MAKSIMUM DOĞRULUK için optimize edildi!")