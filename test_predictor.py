# Geliştirilmiş test scripti - test_predictor_fixed.py olarak kaydet

import os
import sys
import random
from predict import FruitPredictor
import config

def test_fruit_predictor():
    """Test fruit predictor with your trained model"""
    
    # Configuration
    config_dict = config.get_config()
    
    # Model path (adjust to your actual model)
    model_path = 'models/fruits_resnet50_optimal_20250801_010023.h5'  # En iyi modeli kullan
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("Mevcut modeller:")
        models_dir = 'models'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.h5'):
                    print(f"  - {file}")
        return
    
    # Create predictor
    print("🤖 Fruit Predictor başlatılıyor...")
    predictor = FruitPredictor(model_path)
    
    # Load model and class names
    predictor.load_model()
    predictor.load_class_names()
    
    print(f"\n📋 Yüklenen sınıflar ({len(predictor.class_names)}):")
    for i, class_name in enumerate(predictor.class_names[:10]):  # İlk 10'unu göster
        print(f"  {i}: {class_name}")
    if len(predictor.class_names) > 10:
        print(f"  ... ve {len(predictor.class_names) - 10} tane daha")
    
    # Test different modes
    while True:
        print("\n" + "="*50)
        print("🍎 FRUIT PREDICTOR TEST MENÜSÜ")
        print("="*50)
        print("1. Tek görüntü tahmin et")
        print("2. Klasördeki tüm görüntüleri tahmin et") 
        print("3. Kamera ile canlı tahmin")
        print("4. Random test görüntüsü seç")  # ← Değiştirildi
        print("5. Çıkış")
        
        choice = input("\nSeçiminizi yapın (1-5): ").strip()
        
        if choice == '1':
            # Single image prediction
            image_path = input("Görüntü dosya yolunu girin: ").strip()
            if os.path.exists(image_path):
                try:
                    predictions = predictor.predict_single_image(image_path, top_k=5, show_image=True)
                    print("\n✅ Tahmin tamamlandı!")
                except Exception as e:
                    print(f"❌ Hata: {e}")
            else:
                print(f"❌ Dosya bulunamadı: {image_path}")
        
        elif choice == '2':
            # Batch prediction
            folder_path = input("Görüntü klasörü yolunu girin: ").strip()
            if os.path.exists(folder_path):
                try:
                    # Find image files
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    image_paths = []
                    
                    for file in os.listdir(folder_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_paths.append(os.path.join(folder_path, file))
                    
                    if image_paths:
                        print(f"📁 {len(image_paths)} görüntü bulundu")
                        batch_results = predictor.predict_batch_images(image_paths, top_k=3)
                        predictor.create_prediction_gallery(batch_results)
                        print("✅ Batch tahmin tamamlandı!")
                    else:
                        print("❌ Klasörde görüntü dosyası bulunamadı")
                        
                except Exception as e:
                    print(f"❌ Hata: {e}")
            else:
                print(f"❌ Klasör bulunamadı: {folder_path}")
        
        elif choice == '3':
            # Camera prediction
            try:
                print("📹 Kamera başlatılıyor...")
                print("Kontroller: 'q' = çıkış, 's' = fotoğraf çek")
                predictor.predict_from_camera(camera_index=0, continuous=True)
            except Exception as e:
                print(f"❌ Kamera hatası: {e}")
        
        elif choice == '4':
            # RANDOM test image - FIXED VERSION
            test_data_path = config_dict['data'].get('test_path', 'test_data')
            if os.path.exists(test_data_path):
                print(f"📁 Test klasörü: {test_data_path}")
                # List available classes
                classes = [d for d in os.listdir(test_data_path) 
                          if os.path.isdir(os.path.join(test_data_path, d))]
                
                if classes:
                    # RANDOM CLASS SEÇ
                    random_class = random.choice(classes)
                    print(f"🎲 Random seçilen sınıf: {random_class}")
                    
                    # Pick a random image from selected class
                    class_path = os.path.join(test_data_path, random_class)
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if images:
                        # RANDOM IMAGE SEÇ
                        random_image = random.choice(images)
                        test_image = os.path.join(class_path, random_image)
                        print(f"🧪 Test görüntüsü: {test_image}")
                        
                        try:
                            predictions = predictor.predict_single_image(test_image, top_k=5, show_image=True)
                            print(f"🎯 Gerçek sınıf: {random_class}")
                            print(f"🤖 Tahmin: {predictions[0]['class']}")
                            print(f"🔢 Güven: {predictions[0]['percentage']:.2f}%")
                            
                            if predictions[0]['class'] == random_class:
                                print("✅ Doğru tahmin!")
                            else:
                                print("❌ Yanlış tahmin!")
                                
                        except Exception as e:
                            print(f"❌ Hata: {e}")
                    else:
                        print("❌ Test sınıfında görüntü bulunamadı")
                else:
                    print("❌ Test klasöründe sınıf bulunamadı")
            else:
                print(f"❌ Test klasörü bulunamadı: {test_data_path}")
        
        elif choice == '5':
            print("👋 Çıkılıyor...")
            break
        
        else:
            print("❌ Geçersiz seçim! Lütfen 1-5 arası bir sayı girin.")

if __name__ == "__main__":
    test_fruit_predictor()