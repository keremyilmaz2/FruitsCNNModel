"""
CNN Model Mimarileri
Fruits-360 için farklı CNN model mimarileri
"""
import os  # ← BU SATIRIP EKLEYİN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0, VGG16
import config

class FruitsCNNModels:
    def __init__(self, config_dict):
        self.config = config_dict
        self.input_shape = config_dict['image']['input_shape']
        self.num_classes = config_dict['model']['num_classes']
        self.dropout_rate = config_dict['model']['dropout_rate']
        self.use_batch_norm = config_dict['model']['batch_normalization']
    
    def create_custom_cnn_v1(self):
        """Basit özel CNN modeli"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # İlk konvolüsyon bloğu
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # İkinci konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Üçüncü konvolüsyon bloğu
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_custom_cnn_v2(self):
        """Gelişmiş özel CNN modeli"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # İlk konvolüsyon bloğu
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # İkinci konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Üçüncü konvolüsyon bloğu
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Dördüncü konvolüsyon bloğu
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Fully connected layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_residual_block(self, x, filters, kernel_size=3, stride=1):
        """ResNet tarzı residual blok"""
        shortcut = x
        
        # İlk konvolüsyon
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # İkinci konvolüsyon
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            if self.use_batch_norm:
                shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    def create_transfer_learning_model_finetune(self, base_model_name='resnet50', fine_tune_layers=50):
        """Fine-tuning ile transfer learning"""
        
        # Base model seçimi (aynı)
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        # Diğer modeller de aynı...
        
        # ⭐ FINE-TUNING KISMI - YENİ:
        # İlk önce tüm katmanları dondur
        base_model.trainable = False
        
        # Sonra son N katmanı unfreeze et
        if fine_tune_layers > 0:
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True
            print(f"🔓 Son {fine_tune_layers} katman fine-tuning için açıldı")
        
        # Daha basit head (overfitting'i önlemek için)
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),  # Daha fazla dropout
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    def create_custom_resnet(self):
        """Özel ResNet benzeri model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # İlk konvolüsyon
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual bloklar
        x = self.create_residual_block(x, 64)
        x = self.create_residual_block(x, 64)
        
        x = self.create_residual_block(x, 128, stride=2)
        x = self.create_residual_block(x, 128)
        
        x = self.create_residual_block(x, 256, stride=2)
        x = self.create_residual_block(x, 256)
        
        x = self.create_residual_block(x, 512, stride=2)
        x = self.create_residual_block(x, 512)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def create_transfer_learning_model(self, base_model_name='resnet50', freeze_base=True):
        """Transfer learning ile önceden eğitilmiş model kullan"""
        
        # Base model seçimi
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'efficientnetb0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Desteklenmeyen base model: {base_model_name}")
        
        # Base model'i dondur
        if freeze_base:
            base_model.trainable = False
        
        # Özel classification head ekle
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
            layers.Dense(512, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_ensemble_model(self, models_list):
        """Ensemble model oluştur"""
        if len(models_list) < 2:
            raise ValueError("Ensemble için en az 2 model gerekli")
        
        # Her modelin çıktısını al
        model_outputs = []
        for model in models_list:
            model_outputs.append(model.output)
        
        # Ortalama al
        ensemble_output = layers.Average()(model_outputs)
        
        # Ensemble model oluştur
        ensemble_model = models.Model(
            inputs=[model.input for model in models_list],
            outputs=ensemble_output
        )
        
        return ensemble_model
    
    def create_model(self, architecture='custom_cnn_v1'):
        """Belirtilen mimariye göre model oluştur"""
        
        print(f"Model oluşturuluyor: {architecture}")
        
        if architecture == 'custom_cnn_v1':
            model = self.create_custom_cnn_v1()
        elif architecture == 'custom_cnn_v2':
            model = self.create_custom_cnn_v2()
        elif architecture == 'custom_resnet':
            model = self.create_custom_resnet()
        elif architecture == 'transfer_resnet50_finetune':  # ⭐ YENİ SATIRIP
            model = self.create_transfer_learning_model_finetune('resnet50', 50)    
        elif architecture.startswith('transfer_'):
            base_model_name = architecture.replace('transfer_', '')
            model = self.create_transfer_learning_model(base_model_name)
        else:
            raise ValueError(f"Desteklenmeyen mimari: {architecture}")
        
        print(f"Model oluşturuldu!")
        print(f"- Parametre sayısı: {model.count_params():,}")
        print(f"- Input shape: {self.input_shape}")
        print(f"- Output classes: {self.num_classes}")
        
        return model
    
    def compile_model(self, model, optimizer=None, loss=None, metrics=None):
        """Modeli compile et"""
        
        # Varsayılan değerler
        if optimizer is None:
            if self.config['training']['optimizer'] == 'adam':
                optimizer = keras.optimizers.Adam(
                    learning_rate=self.config['training']['learning_rate']
                )
            elif self.config['training']['optimizer'] == 'sgd':
                optimizer = keras.optimizers.SGD(
                    learning_rate=self.config['training']['learning_rate'],
                    momentum=0.9
                )
            else:
                optimizer = self.config['training']['optimizer']
        
        if loss is None:
            loss = self.config['training']['loss_function']
        
        if metrics is None:
            metrics = self.config['training']['metrics']
        
        # Model'i compile et
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("Model compile edildi!")
        return model
    
    def get_callbacks(self):
        """Eğitim için callback'leri hazırla"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoints_dir'],
            f"best_model_{self.config['model']['model_name']}.h5"
        )
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config['save']['checkpoint_monitor'],
            mode=self.config['save']['checkpoint_mode'],
            save_best_only=self.config['training']['save_best_only'],
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Reduce learning rate
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        # # TensorBoard
        # log_dir = os.path.join(self.config['paths']['logs_dir'], 
        #                       f"{self.config['model']['model_name']}")
        # tensorboard_callback = keras.callbacks.TensorBoard(
        #     log_dir=log_dir,
        #     histogram_freq=1,
        #     write_graph=True,
        #     write_images=True
        # )
        # callbacks.append(tensorboard_callback)
        
        return callbacks

def create_model_builder(config_dict=None):
    """FruitsCNNModels örneği oluştur"""
    if config_dict is None:
        config_dict = config.get_config()
    
    return FruitsCNNModels(config_dict)

if __name__ == "__main__":
    # Test amaçlı kullanım
    import os
    
    config_dict = config.get_config()
    model_builder = FruitsCNNModels(config_dict)
    
    # Farklı modelleri test et
    architectures = ['custom_cnn_v1', 'custom_cnn_v2', 'transfer_resnet50']
    
    for arch in architectures:
        try:
            print(f"\n{'='*50}")
            print(f"Test ediliyor: {arch}")
            print('='*50)
            
            if arch.startswith('transfer_'):
                model = model_builder.create_transfer_learning_model(
                    arch.replace('transfer_', '')
                )
            else:
                model = model_builder.create_model(arch)
            
            model = model_builder.compile_model(model)
            
            if config_dict['debug']['print_model_summary']:
                model.summary()
            
            print(f"✅ {arch} başarılı!")
            
        except Exception as e:
            print(f"❌ {arch} hatası: {e}")