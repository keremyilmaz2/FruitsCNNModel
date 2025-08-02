"""
Fruits-360 CNN Model Eğitimi
Ana eğitim dosyası - Keras/TensorFlow ile meyve tanıma modeli
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Kendi modüllerimizi import et
import config
import utils
import data_preprocessing
import model_architecture
from tensorflow import keras
def main():
    """Ana eğitim fonksiyonu"""
    
    print("🍎 Fruits-360 CNN Model Eğitimi Başlıyor 🍎")
    print("=" * 60)
    
    # Konfigürasyonu yükle
    config_dict = config.get_config()
    
    # Gerekli klasörleri oluştur
    config.create_directories()
    
    # GPU ayarlarını yap
    utils.setup_gpu()
    
    print(f"Eğitim parametreleri:")
    print(f"- Batch size: {config_dict['training']['batch_size']}")
    print(f"- Epochs: {config_dict['training']['epochs']}")
    print(f"- Learning rate: {config_dict['training']['learning_rate']}")
    print(f"- Model: {config_dict['model']['architecture']}")
    print(f"- Optimizer: {config_dict['training']['optimizer']}")
    print()
    
    # Veri işleyiciyi oluştur
    print("📊 Veri işleme başlıyor...")
    data_processor = data_preprocessing.create_data_processor(config_dict)
    
    # Veri setini analiz et
    train_path = config_dict['data']['train_path']
    if not os.path.exists(train_path):
        print(f"❌ Hata: Training klasörü bulunamadı: {train_path}")
        print("Lütfen config.py dosyasındaki veri yollarını kontrol edin.")
        return
    
    print("🔍 Veri seti analiz ediliyor...")
    class_counts = data_processor.analyze_dataset(train_path)
    
    # Veri setini yükle
    method = input("\nVeri yükleme yöntemi seçin:\n1. ImageDataGenerator (Önerilen - Bellek dostu)\n2. Tüm veriyi belleğe yükle\nSeçiminiz (1 veya 2): ").strip()
    
    if method == "2":
        print("💾 Tüm veri seti belleğe yükleniyor...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_full_dataset(
            train_path=config_dict['data']['train_path'],
            test_path=config_dict['data']['test_path'],
            validation_path=None,  # Training'den split yapacak
            save_preprocessed=True
        )
        
        train_generator = None
        validation_generator = None
        test_generator = None
        
        steps_per_epoch = len(X_train) // config_dict['training']['batch_size']
        validation_steps = len(X_val) // config_dict['training']['batch_size']
        
    else:
        print("🔄 ImageDataGenerator ile veri generatörleri oluşturuluyor...")
        
        # Validation klasörü yoksa oluştur
        val_path = config_dict['data']['train_path'].replace('Training', 'Validation')
        if not os.path.exists(val_path):
            print("📁 Validation klasörü oluşturuluyor...")
            data_processor.create_validation_split(
                config_dict['data']['train_path'], 
                val_path, 
                config_dict['training']['validation_split']
            )
        
        # Veri generatörlerini oluştur
        train_generator, validation_generator, test_generator = data_processor.create_data_generators(
            train_path=config_dict['data']['train_path'],
            validation_path=val_path,
            test_path=config_dict['data']['test_path']
        )
        
        steps_per_epoch = train_generator.samples // config_dict['training']['batch_size']
        validation_steps = validation_generator.samples // config_dict['training']['batch_size']
        
        X_train, y_train = None, None
        X_val, y_val = None, None
        X_test, y_test = None, None
    
    # Model oluşturucu
    print("🏗️ Model oluşturuluyor...")
    model_builder = model_architecture.create_model_builder(config_dict)
    
    # Model mimarisini seç
    architecture = config_dict['model']['architecture']
    model = model_builder.create_model(architecture)
    
    # Modeli compile et
    # Modeli compile et
    model = model_builder.compile_model(model)
    
    # ✅ BURAYA TAŞıYıN:
    if config_dict['model']['architecture'] == 'transfer_resnet50_finetune':
        # Önceki modelin ağırlıklarını yükle
        previous_model_path = 'models/fruits_resnet50_optimal_20250801_010023.h5'
        if os.path.exists(previous_model_path):
            print("🔄 Önceki model ağırlıkları yükleniyor...")
            previous_model = keras.models.load_model(previous_model_path)
            
            # Sadece base model ağırlıklarını transfer et
            for i, layer in enumerate(model.layers[0].layers):  # ResNet50 katmanları
                if i < len(previous_model.layers[0].layers):
                    layer.set_weights(previous_model.layers[0].layers[i].get_weights())
            
            print("✅ Önceki model ağırlıkları transfer edildi!")
    
    # Model özetini yazdır
    if config_dict['debug']['print_model_summary']:
        print("\n📋 Model Özeti:")
        model.summary()
        print()
    
    # Model özetini yazdır
    if config_dict['debug']['print_model_summary']:
        print("\n📋 Model Özeti:")
        model.summary()
        print()
    
    # Callback'leri hazırla
    callbacks = model_builder.get_callbacks()
    
    # Sınıf ağırlıklarını hesapla (opsiyonel)
    class_weights = None
    if method == "2" and len(set(class_counts.values())) > 1:  # Dengesiz veri varsa
        print("⚖️ Sınıf ağırlıkları hesaplanıyor...")
        # y_train'den string label'ları çıkar
        string_labels = [data_processor.class_names[np.argmax(label)] for label in y_train]
        class_weights = data_processor.get_class_weights(string_labels)
    
    print("🚀 Model eğitimi başlıyor...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print()
    
    # Eğitim zamanını başlat
    start_time = datetime.now()
    
    # Model eğitimi
    if method == "2":  # Bellek metodu
        history = model.fit(
            X_train, y_train,
            batch_size=config_dict['training']['batch_size'],
            epochs=config_dict['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=config_dict['debug']['verbose'],
            shuffle=config_dict['training']['shuffle']
        )
    else:  # Generator metodu
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config_dict['training']['epochs'],
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=config_dict['debug']['verbose']
        )
    
    # Eğitim süresini hesapla
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"⏱️ Toplam süre: {training_duration}")
    print(f"📊 En iyi validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    # Modeli kaydet
    if config_dict['save']['save_model']:
        print("💾 Model kaydediliyor...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{config_dict['model']['model_name']}_{timestamp}"
        
        # Model dosya yolu
        model_path = os.path.join(
            config_dict['paths']['models_dir'], 
            f"{model_name}.h5"
        )
        
        # Modeli kaydet
        model.save(model_path)
        print(f"✅ Model kaydedildi: {model_path}")
        
        # Model boyutunu göster
        model_size = utils.calculate_model_size(model_path)
        print(f"📏 Model boyutu: {model_size}")
    
    # Eğitim geçmişini görselleştir
    if config_dict['visualization']['plot_history']:
        print("📈 Eğitim grafiği oluşturuluyor...")
        history_plot_path = os.path.join(
            config_dict['paths']['results_dir'],
            f"training_history_{timestamp}.png"
        )
        utils.plot_training_history(history, history_plot_path)
    
    # Model bilgilerini kaydet
    print("📄 Model bilgileri kaydediliyor...")
    utils.save_model_info(
        model, history, config_dict, 
        config_dict['paths']['results_dir']
    )
    
    # Test seti değerlendirmesi
    if (method == "2" and X_test is not None) or (method == "1" and test_generator is not None):
        print("🧪 Test seti üzerinde değerlendirme yapılıyor...")
        
        if method == "2":
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Tahminler yap
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
        else:
            test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Tahminler yap
            test_generator.reset()
            y_pred = model.predict(test_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = test_generator.classes
        
        # Classification report
        if config_dict['visualization']['plot_confusion_matrix']:
            print("📊 Classification report oluşturuluyor...")
            report_path = os.path.join(
                config_dict['paths']['results_dir'],
                f"classification_report_{timestamp}.txt"
            )
            utils.print_classification_report(
                y_true_classes, y_pred_classes, 
                data_processor.class_names, report_path
            )
            
            # Confusion matrix
            print("🎯 Confusion matrix oluşturuluyor...")
            cm_path = os.path.join(
                config_dict['paths']['results_dir'],
                f"confusion_matrix_{timestamp}.png"
            )
            utils.plot_confusion_matrix(
                y_true_classes, y_pred_classes,
                data_processor.class_names, cm_path
            )
    
    # Örnek tahminler göster
    if config_dict['visualization']['plot_sample_predictions']:
        print("🔍 Örnek tahminler gösteriliyor...")
        show_sample_predictions(
            model, data_processor, config_dict,
            method, X_test, test_generator, timestamp
        )
    
    print("\n🎉 Tüm işlemler başarıyla tamamlandı!")
    print(f"📁 Sonuçlar klasörü: {config_dict['paths']['results_dir']}")
    print(f"📁 Model klasörü: {config_dict['paths']['models_dir']}")


def show_sample_predictions(model, data_processor, config_dict, method, X_test, test_generator, timestamp):
    """Örnek tahminleri göster"""
    import matplotlib.pyplot as plt
    
    samples_to_show = min(config_dict['visualization']['samples_to_plot'], 20)
    
    if method == "2" and X_test is not None:
        # Rastgele örnekler seç
        random_indices = np.random.choice(len(X_test), samples_to_show, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, idx in enumerate(random_indices):
            img = X_test[idx]
            true_class_idx = np.argmax(y_test[idx])
            true_class = data_processor.class_names[true_class_idx]
            
            # Tahmin yap
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            pred_class_idx = np.argmax(pred)
            pred_class = data_processor.class_names[pred_class_idx]
            confidence = pred[pred_class_idx]
            
            # Görüntüyü göster
            axes[i].imshow(img)
            axes[i].set_title(f'Gerçek: {true_class}\nTahmin: {pred_class}\nGüven: {confidence:.2f}',
                            color='green' if true_class == pred_class else 'red')
            axes[i].axis('off')
    
    elif method == "1" and test_generator is not None:
        # Generator'den örnekler al
        test_generator.reset()
        batch_images, batch_labels = next(test_generator)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()
        
        for i in range(min(samples_to_show, len(batch_images))):
            img = batch_images[i]
            true_class_idx = np.argmax(batch_labels[i])
            true_class = data_processor.class_names[true_class_idx]
            
            # Tahmin yap
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            pred_class_idx = np.argmax(pred)
            pred_class = data_processor.class_names[pred_class_idx]
            confidence = pred[pred_class_idx]
            
            # Görüntüyü göster
            axes[i].imshow(img)
            axes[i].set_title(f'Gerçek: {true_class}\nTahmin: {pred_class}\nGüven: {confidence:.2f}',
                            color='green' if true_class == pred_class else 'red')
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Grafiği kaydet
    if config_dict['visualization']['save_plots']:
        predictions_path = os.path.join(
            config_dict['paths']['results_dir'],
            f"sample_predictions_{timestamp}.png"
        )
        plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
        print(f"✅ Örnek tahminler kaydedildi: {predictions_path}")
    
    plt.show()


def interactive_training():
    """İnteraktif eğitim modu"""
    print("🎛️ İnteraktif Eğitim Modu")
    print("=" * 40)
    
    # Konfigürasyonu yükle
    config_dict = config.get_config()
    
    # Kullanıcıdan parametreleri al
    print("Eğitim parametrelerini özelleştirmek ister misiniz? (y/n): ", end="")
    customize = input().strip().lower()
    
    if customize == 'y':
        # Epochs
        epochs = input(f"Epochs ({config_dict['training']['epochs']}): ").strip()
        if epochs:
            config_dict['training']['epochs'] = int(epochs)
        
        # Batch size
        batch_size = input(f"Batch size ({config_dict['training']['batch_size']}): ").strip()
        if batch_size:
            config_dict['training']['batch_size'] = int(batch_size)
        
        # Learning rate
        lr = input(f"Learning rate ({config_dict['training']['learning_rate']}): ").strip()
        if lr:
            config_dict['training']['learning_rate'] = float(lr)
        
        # Model mimarisi
        print("Mevcut mimariler: custom_cnn_v1, custom_cnn_v2, transfer_resnet50, transfer_mobilenetv2")
        arch = input(f"Mimari ({config_dict['model']['architecture']}): ").strip()
        if arch:
            config_dict['model']['architecture'] = arch
        
        # Data augmentation
        aug = input(f"Data augmentation kullan? (y/n) ({config_dict['augmentation']['use_augmentation']}): ").strip()
        if aug:
            config_dict['augmentation']['use_augmentation'] = aug.lower() == 'y'
    
    print("\n🚀 Eğitim başlatılıyor...")
    return config_dict


if __name__ == "__main__":
    try:
        # Komut satırı argümanlarını kontrol et
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            config_dict = interactive_training()
            # Global config'i güncelle
            for key, value in config_dict.items():
                if hasattr(config, key.upper() + '_CONFIG'):
                    getattr(config, key.upper() + '_CONFIG').update(value)
        
        # Ana eğitim fonksiyonunu çalıştır
        main()
        
    except KeyboardInterrupt:
        print("\n⏹️ Eğitim kullanıcı tarafından durduruldu!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        print("Lütfen config.py dosyasındaki ayarları ve veri yollarını kontrol edin.")
        sys.exit(1)