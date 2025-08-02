"""
Kurulum ve Veri Seti Kontrolü
Projeyi çalıştırmadan önce tüm gereksinimleri kontrol eder
"""

import os
import sys
import importlib.util

def check_python_version():
    """Python sürümünü kontrol et"""
    print("🐍 Python Sürümü Kontrolü:")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("   ❌ Python 3.7+ gerekli!")
        return False
    else:
        print("   ✅ Python sürümü uygun")
        return True

def check_required_packages():
    """Gerekli paketleri kontrol et"""
    print("\n📦 Gerekli Paketler Kontrolü:")
    
    required_packages = {
        'tensorflow': 'tensorflow',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn', 
        'scikit-learn': 'sklearn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                print(f"   ✅ {package_name}")
            else:
                print(f"   ❌ {package_name} - Bulunamadı")
                missing_packages.append(package_name)
        except ImportError:
            print(f"   ❌ {package_name} - Import hatası")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n🔧 Eksik paketleri yüklemek için:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_tensorflow_gpu():
    """TensorFlow GPU desteğini kontrol et"""
    print("\n🎮 TensorFlow GPU Kontrolü:")
    
    try:
        import tensorflow as tf
        print(f"   TensorFlow sürümü: {tf.__version__}")
        
        # GPU kontrolü
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ {len(gpus)} GPU bulundu:")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   ⚠️ GPU bulunamadı, CPU kullanılacak")
        
        return True
    except ImportError:
        print("   ❌ TensorFlow yüklü değil!")
        return False

def check_dataset_structure():
    """Veri seti yapısını kontrol et"""
    print("\n📁 Veri Seti Yapısı Kontrolü:")
    
    # Config'i import et
    try:
        import config
        config_dict = config.get_config()
    except ImportError:
        print("   ❌ config.py dosyası bulunamadı!")
        return False
    
    # Veri yollarını kontrol et
    dataset_path = config_dict['data']['dataset_path']
    train_path = config_dict['data']['train_path']
    test_path = config_dict['data']['test_path']
    
    print(f"   Dataset path: {dataset_path}")
    
    # Ana klasör kontrolü
    if not os.path.exists(dataset_path):
        print(f"   ❌ Ana veri klasörü bulunamadı: {dataset_path}")
        return False
    else:
        print(f"   ✅ Ana veri klasörü mevcut")
    
    # Training klasörü kontrolü
    if not os.path.exists(train_path):
        print(f"   ❌ Training klasörü bulunamadı: {train_path}")
        return False
    else:
        print(f"   ✅ Training klasörü mevcut")
        
        # Sınıf sayısını kontrol et
        class_folders = [d for d in os.listdir(train_path) 
                        if os.path.isdir(os.path.join(train_path, d))]
        print(f"   📊 Training'de {len(class_folders)} sınıf bulundu")
        
        if len(class_folders) < 50:
            print("   ⚠️ Sınıf sayısı beklenenden az, veri seti tam değil olabilir")
    
    # Test klasörü kontrolü  
    if not os.path.exists(test_path):
        print(f"   ❌ Test klasörü bulunamadı: {test_path}")
        return False
    else:
        print(f"   ✅ Test klasörü mevcut")
        
        # Test sınıf sayısını kontrol et
        test_class_folders = [d for d in os.listdir(test_path) 
                             if os.path.isdir(os.path.join(test_path, d))]
        print(f"   📊 Test'te {len(test_class_folders)} sınıf bulundu")
    
    return True

def check_project_structure():
    """Proje dosyalarını kontrol et"""
    print("\n🏗️ Proje Yapısı Kontrolü:")
    
    required_files = [
        'config.py',
        'utils.py', 
        'data_preprocessing.py',
        'model_architecture.py',
        'train_fruits_cnn.py',
        'evaluate_model.py',
        'predict.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Bulunamadı")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Eksik dosyalar: {', '.join(missing_files)}")
        return False
    
    return True

def create_required_directories():
    """Gerekli klasörleri oluştur"""
    print("\n📂 Gerekli Klasörler Oluşturuluyor:")
    
    try:
        import config
        config.create_directories()
        print("   ✅ Tüm klasörler oluşturuldu")
        return True
    except Exception as e:
        print(f"   ❌ Klasör oluşturma hatası: {e}")
        return False

def sample_data_check():
    """Örnek veri kontrolü"""
    print("\n🔍 Örnek Veri Kontrolü:")
    
    try:
        import config
        config_dict = config.get_config()
        train_path = config_dict['data']['train_path']
        
        # İlk sınıftan bir örnek al
        class_folders = [d for d in os.listdir(train_path) 
                        if os.path.isdir(os.path.join(train_path, d))]
        
        if class_folders:
            first_class = class_folders[0]
            first_class_path = os.path.join(train_path, first_class)
            
            # Görüntü dosyalarını kontrol et
            image_files = [f for f in os.listdir(first_class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                print(f"   ✅ Örnek sınıf: {first_class}")
                print(f"   📊 {len(image_files)} görüntü bulundu")
                
                # İlk görüntüyü kontrol et
                first_image = os.path.join(first_class_path, image_files[0])
                
                try:
                    import cv2
                    img = cv2.imread(first_image)
                    if img is not None:
                        print(f"   ✅ Görüntü okunabilir: {img.shape}")
                        return True
                    else:
                        print(f"   ❌ Görüntü okunamadı: {first_image}")
                        return False
                except ImportError:
                    print("   ⚠️ OpenCV yüklü değil, görüntü kontrolü atlandı")
                    return True
            else:
                print(f"   ❌ {first_class} klasöründe görüntü bulunamadı")
                return False
        else:
            print("   ❌ Hiç sınıf klasörü bulunamadı")
            return False
            
    except Exception as e:
        print(f"   ❌ Örnek veri kontrolü hatası: {e}")
        return False

def main():
    """Ana kontrol fonksiyonu"""
    print("🚀 Fruits-360 CNN Proje Kurulum Kontrolü")
    print("=" * 60)
    
    checks = [
        ("Python Sürümü", check_python_version),
        ("Gerekli Paketler", check_required_packages),
        ("TensorFlow GPU", check_tensorflow_gpu),
        ("Proje Yapısı", check_project_structure),
        ("Veri Seti Yapısı", check_dataset_structure),
        ("Gerekli Klasörler", create_required_directories),
        ("Örnek Veri", sample_data_check)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ {check_name} kontrolü sırasında hata: {e}")
            results.append((check_name, False))
    
    # Özet
    print("\n" + "=" * 60)
    print("📋 KONTROL ÖZETİ:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{check_name:<20} : {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Sonuç: {passed}/{total} kontrol başarılı")
    
    if passed == total:
        print("\n🎉 Tüm kontroller başarılı! Projeyi çalıştırabilirsiniz.")
        print("\n🚀 Eğitime başlamak için:")
        print("   python train_fruits_cnn.py")
        return True
    else:
        print(f"\n⚠️ {total-passed} kontrol başarısız. Lütfen sorunları çözün.")
        print("\n🔧 Sık karşılaşılan çözümler:")
        print("   1. pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy tqdm")
        print("   2. Veri seti yollarını config.py'de kontrol edin")
        print("   3. Fruits-360 veri setini doğru klasöre çıkarın")
        return False

if __name__ == "__main__":
    main()