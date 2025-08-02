"""
Veri Ön İşleme Modülü
Fruits-360 veri seti için veri yükleme, augmentation ve preprocessing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import config
import utils

class FruitsDataProcessor:
    def __init__(self, config_dict):
        self.config = config_dict
        self.class_names = []
        self.label_encoder = LabelEncoder()
        self.image_size = config_dict['image']['image_size']
        self.num_classes = config_dict['model']['num_classes']
        
    def load_data_from_directory(self, data_path, subset='training'):
        """Klasörlerden veri yükle"""
        print(f"{subset} verisi yükleniyor: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_path}")
        
        # Sınıf isimlerini al
        if not self.class_names:
            self.class_names = utils.get_class_names(data_path)
            print(f"Toplam {len(self.class_names)} sınıf bulundu")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(tqdm(self.class_names, desc="Sınıflar işleniyor")):
            class_path = os.path.join(data_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Uyarı: {class_name} klasörü bulunamadı")
                continue
            
            # Klasördeki görüntü dosyalarını al
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Görüntüyü yükle ve işle
                    img = self.load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Görüntü yükleme hatası {img_path}: {e}")
                    continue
        
        print(f"{subset} - Toplam {len(images)} görüntü yüklendi")
        
        # NumPy array'lere çevir
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        return X, y
    
    def load_and_preprocess_image(self, img_path):
        """Tek bir görüntüyü yükle ve önişle"""
        try:
            # OpenCV ile görüntüyü yükle
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # BGR'den RGB'ye çevir
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Boyutu yeniden ayarla
            img = cv2.resize(img, self.image_size)
            
            # Normalizasyon (0-1 arası)
            if self.config['image']['normalization']:
                img = img.astype(np.float32) / 255.0
            
            return img
        
        except Exception as e:
            print(f"Görüntü işleme hatası: {e}")
            return None
    
    def encode_labels(self, labels):
        """String label'ları sayısal değerlere çevir"""
        if not hasattr(self.label_encoder, 'classes_'):
            # İlk kez fit ediyoruz
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            # Daha önce fit edilmiş
            encoded_labels = self.label_encoder.transform(labels)
        
        # One-hot encoding
        categorical_labels = to_categorical(encoded_labels, num_classes=len(self.class_names))
        
        return encoded_labels, categorical_labels
    
    def create_data_generators(self, train_path, validation_path=None, test_path=None):
        """Keras ImageDataGenerator ile veri generatörleri oluştur"""
        
        # Training data generator (with augmentation)
        if self.config['augmentation']['use_augmentation']:
            train_datagen = ImageDataGenerator(
                rescale=1./255 if self.config['image']['normalization'] else 1.0,
                rotation_range=self.config['augmentation']['rotation_range'],
                width_shift_range=self.config['augmentation']['width_shift_range'],
                height_shift_range=self.config['augmentation']['height_shift_range'],
                shear_range=self.config['augmentation']['shear_range'],
                zoom_range=self.config['augmentation']['zoom_range'],
                horizontal_flip=self.config['augmentation']['horizontal_flip'],
                vertical_flip=self.config['augmentation']['vertical_flip'],
                fill_mode=self.config['augmentation']['fill_mode'],
                validation_split=self.config['training']['validation_split'] if validation_path is None else 0.0
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255 if self.config['image']['normalization'] else 1.0,
                validation_split=self.config['training']['validation_split'] if validation_path is None else 0.0
            )
        
        # Validation/Test data generator (no augmentation)
        val_test_datagen = ImageDataGenerator(
            rescale=1./255 if self.config['image']['normalization'] else 1.0
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.image_size,
            batch_size=self.config['training']['batch_size'],
            class_mode='categorical',
            shuffle=self.config['training']['shuffle'],
            subset='training' if validation_path is None else None
        )
        
        # Validation generator
        if validation_path:
            validation_generator = val_test_datagen.flow_from_directory(
                validation_path,
                target_size=self.image_size,
                batch_size=self.config['training']['batch_size'],
                class_mode='categorical',
                shuffle=False
            )
        else:
            validation_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=self.image_size,
                batch_size=self.config['training']['batch_size'],
                class_mode='categorical',
                shuffle=False,
                subset='validation'
            )
        
        # Test generator
        test_generator = None
        if test_path and os.path.exists(test_path):
            test_generator = val_test_datagen.flow_from_directory(
                test_path,
                target_size=self.image_size,
                batch_size=self.config['training']['batch_size'],
                class_mode='categorical',
                shuffle=False
            )
        
        # Sınıf isimlerini güncelle
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Veri generatörleri oluşturuldu:")
        print(f"- Training samples: {train_generator.samples}")
        print(f"- Validation samples: {validation_generator.samples}")
        if test_generator:
            print(f"- Test samples: {test_generator.samples}")
        print(f"- Sınıf sayısı: {self.num_classes}")
        
        return train_generator, validation_generator, test_generator
    
    def load_full_dataset(self, train_path, test_path, validation_path=None, save_preprocessed=True):
        """Tüm veri setini belleğe yükle"""
        
        # Training verisi yükle
        X_train, y_train = self.load_data_from_directory(train_path, 'training')
        
        # Test verisi yükle
        X_test, y_test = self.load_data_from_directory(test_path, 'test')
        
        # Validation verisi yükle (varsa)
        if validation_path and os.path.exists(validation_path):
            X_val, y_val = self.load_data_from_directory(validation_path, 'validation')
        else:
            # Training'den validation oluştur
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config['training']['validation_split'],
                stratify=y_train,
                random_state=42
            )
        
        # Label encoding
        # Tüm label'ları birleştir
        all_labels = np.concatenate([y_train, y_val, y_test])
        self.label_encoder.fit(all_labels)
        self.class_names = list(self.label_encoder.classes_)
        
        # Label'ları encode et
        _, y_train_cat = self.encode_labels(y_train)
        _, y_val_cat = self.encode_labels(y_val)
        _, y_test_cat = self.encode_labels(y_test)
        
        print(f"Veri seti yüklendi:")
        print(f"- Training: {X_train.shape[0]} görüntü")
        print(f"- Validation: {X_val.shape[0]} görüntü")
        print(f"- Test: {X_test.shape[0]} görüntü")
        print(f"- Görüntü boyutu: {X_train.shape[1:]}")
        print(f"- Sınıf sayısı: {len(self.class_names)}")
        
        # Önişlenmiş veriyi kaydet
        if save_preprocessed:
            self.save_preprocessed_data(X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat)
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)
    
    def save_preprocessed_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Önişlenmiş veriyi kaydet"""
        save_dir = self.config['paths']['results_dir']
        
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': self.class_names,
            'label_encoder': self.label_encoder
        }
        
        save_path = os.path.join(save_dir, 'preprocessed_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Önişlenmiş veri kaydedildi: {save_path}")
    
    def load_preprocessed_data(self):
        """Önceden işlenmiş veriyi yükle"""
        save_path = os.path.join(self.config['paths']['results_dir'], 'preprocessed_data.pkl')
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Önişlenmiş veri bulunamadı: {save_path}")
        
        with open(save_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.class_names = data_dict['class_names']
        self.label_encoder = data_dict['label_encoder']
        
        print("Önişlenmiş veri yüklendi!")
        print(f"- Sınıf sayısı: {len(self.class_names)}")
        
        return (data_dict['X_train'], data_dict['y_train']), \
               (data_dict['X_val'], data_dict['y_val']), \
               (data_dict['X_test'], data_dict['y_test'])
    
    def get_class_weights(self, labels):
        """Dengesiz veri için sınıf ağırlıkları hesapla"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # String label'ları sayısal değerlere çevir
        if isinstance(labels[0], str):
            encoded_labels = self.label_encoder.transform(labels)
        else:
            encoded_labels = labels
        
        # Sınıf ağırlıklarını hesapla
        classes = np.unique(encoded_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=encoded_labels
        )
        
        # Sözlük formatında döndür
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Sınıf ağırlıkları hesaplandı: {len(class_weight_dict)} sınıf")
        return class_weight_dict
    def create_validation_split(self, train_path, val_path, split_ratio=0.2):
        """Training setinden validation seti oluştur"""
        import shutil
        
        if os.path.exists(val_path):
            print("Validation klasörü zaten mevcut!")
            return
        
        os.makedirs(val_path, exist_ok=True)
        
        # Sınıf klasörlerini al
        class_names = [d for d in os.listdir(train_path) 
                       if os.path.isdir(os.path.join(train_path, d))]
        
        print(f"🔄 {len(class_names)} sınıf için validation split yapılıyor...")
        
        for class_name in class_names:
            source_class_path = os.path.join(train_path, class_name)
            target_class_path = os.path.join(val_path, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            
            # Sınıf içindeki tüm görüntüleri al
            images = [f for f in os.listdir(source_class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Validation için rastgele seç
            np.random.seed(42)  # Reproducible split
            np.random.shuffle(images)
            val_count = int(len(images) * split_ratio)
            val_images = images[:val_count]
            
            # Validation klasörüne KOPYALA
            for img_name in val_images:
                source = os.path.join(source_class_path, img_name)
                target = os.path.join(target_class_path, img_name)
                try:
                    shutil.copy2(source, target)
                except Exception as e:
                    print(f"Kopyalama hatası {img_name}: {e}")
            
            if len(val_images) > 0:
                print(f"✅ {class_name}: {len(val_images)} görüntü validation'a kopyalandı")
        
        print(f"✅ Validation split tamamlandı: {val_path}")
    def analyze_dataset(self, data_path):
        """Veri setini analiz et"""
        print(f"Veri seti analizi: {data_path}")
        print("=" * 50)
        
        # Sınıf başına görüntü sayıları
        class_counts = utils.count_images_per_class(data_path)
        
        # İstatistikler
        total_images = sum(class_counts.values())
        avg_images_per_class = total_images / len(class_counts)
        min_images = min(class_counts.values())
        max_images = max(class_counts.values())
        
        print(f"Toplam görüntü sayısı: {total_images}")
        print(f"Sınıf sayısı: {len(class_counts)}")
        print(f"Sınıf başına ortalama görüntü: {avg_images_per_class:.2f}")
        print(f"En az görüntüye sahip sınıf: {min_images}")
        print(f"En fazla görüntüye sahip sınıf: {max_images}")
        
        # En az ve en fazla görüntüye sahip sınıfları göster
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
        print(f"\nEn az görüntüye sahip 5 sınıf:")
        for class_name, count in sorted_counts[:5]:
            print(f"  {class_name}: {count}")
        
        print(f"\nEn fazla görüntüye sahip 5 sınıf:")
        for class_name, count in sorted_counts[-5:]:
            print(f"  {class_name}: {count}")
        
        return class_counts

def create_data_processor(config_dict=None):
    """FruitsDataProcessor örneği oluştur"""
    if config_dict is None:
        config_dict = config.get_config()
    
    return FruitsDataProcessor(config_dict)

if __name__ == "__main__":
    # Test amaçlı kullanım
    config_dict = config.get_config()
    processor = FruitsDataProcessor(config_dict)
    
    # Veri seti analizi
    train_path = config_dict['data']['train_path']
    if os.path.exists(train_path):
        class_counts = processor.analyze_dataset(train_path)
        
        # Örnek görüntüler göster
        utils.display_sample_images(train_path, processor.class_names)
        
        # Veri dağılımını görselleştir
        utils.plot_data_distribution(class_counts, 
                                   save_path=os.path.join(config_dict['paths']['results_dir'], 
                                                         'data_distribution.png'))
    else:
        print(f"Veri klasörü bulunamadı: {train_path}")
        print("Lütfen config.py dosyasındaki veri yollarını kontrol edin.")