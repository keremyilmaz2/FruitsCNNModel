"""
Tahmin Modülü
Eğitilmiş model ile yeni görüntüler üzerinde tahmin yapma
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import json
from datetime import datetime
import argparse

import config
import utils

class FruitPredictor:
    def __init__(self, model_path, class_names=None):
        self.model_path = model_path
        self.model = None
        self.class_names = class_names or []
        self.config = config.get_config()
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        
        print(f"Model yükleniyor: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model başarıyla yüklendi!")
        
        # Model bilgilerini göster
        print(f"📏 Model boyutu: {utils.calculate_model_size(self.model_path)}")
        print(f"📊 Input shape: {self.model.input_shape}")
        print(f"🎯 Output classes: {self.model.output_shape[1]}")
        
        return self.model
    
    def load_class_names(self, class_names_path=None):
        """Sınıf isimlerini yükle"""
        if class_names_path and os.path.exists(class_names_path):
            # JSON dosyasından yükle
            with open(class_names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.class_names = data.get('class_names', [])
        elif not self.class_names:
            # Training klasöründen otomatik al
            train_path = self.config['data']['train_path']
            if os.path.exists(train_path):
                self.class_names = utils.get_class_names(train_path)
            else:
                print("⚠️ Uyarı: Sınıf isimleri bulunamadı!")
                self.class_names = [f'Class_{i}' for i in range(self.model.output_shape[1])]
        
        print(f"📋 {len(self.class_names)} sınıf yüklendi")
        return self.class_names
    
    def preprocess_image(self, image_path):
        """Görüntüyü model için hazırla"""
        if isinstance(image_path, str):
            # Dosya yolundan yükle
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Görüntü okunamadı: {image_path}")
                
        else:
            # Numpy array olarak verilmiş
            img = image_path
        
        # BGR'den RGB'ye çevir
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Boyutu ayarla
        target_size = self.config['image']['image_size']
        img = cv2.resize(img, target_size)
        
        # Normalizasyon
        if self.config['image']['normalization']:
            img = img.astype(np.float32) / 255.0
        
        # Batch dimension ekle
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_single_image(self, image_path, top_k=5, show_image=True):
        """Tek bir görüntü için tahmin yap"""
        if self.model is None:
            self.load_model()
        
        if not self.class_names:
            self.load_class_names()
        
        # Görüntüyü hazırla
        processed_img = self.preprocess_image(image_path)
        
        # Tahmin yap
        predictions = self.model.predict(processed_img, verbose=0)[0]
        
        # En yüksek k tahmini al
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []
        
        for i, idx in enumerate(top_indices):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f'Class_{idx}'
            confidence = predictions[idx]
            top_predictions.append({
                'rank': i + 1,
                'class': class_name,
                'confidence': float(confidence),
                'percentage': float(confidence * 100)
            })
        
        # Sonuçları göster
        print(f"\n🔍 Tahmin Sonuçları: {os.path.basename(image_path) if isinstance(image_path, str) else 'Image'}")
        print("-" * 50)
        for pred in top_predictions:
            print(f"{pred['rank']}. {pred['class']:<20} {pred['percentage']:>6.2f}%")
        
        # Görüntüyü göster
        if show_image and isinstance(image_path, str):
            self.show_prediction_result(image_path, top_predictions)
        
        return top_predictions
    
    def predict_batch_images(self, image_paths, top_k=3, save_results=True):
        """Birden fazla görüntü için tahmin yap"""
        if self.model is None:
            self.load_model()
        
        if not self.class_names:
            self.load_class_names()
        
        batch_results = []
        
        print(f"📸 {len(image_paths)} görüntü işleniyor...")
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"İşleniyor ({i+1}/{len(image_paths)}): {os.path.basename(image_path)}")
                
                predictions = self.predict_single_image(
                    image_path, top_k=top_k, show_image=False
                )
                
                batch_results.append({
                    'image_path': image_path,
                    'filename': os.path.basename(image_path),
                    'predictions': predictions,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"❌ Hata ({image_path}): {e}")
                batch_results.append({
                    'image_path': image_path,
                    'filename': os.path.basename(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Sonuçları kaydet
        if save_results:
            self.save_batch_results(batch_results)
        
        return batch_results
    
    def predict_from_camera(self, camera_index=0, continuous=True):
        """Kameradan canlı tahmin yap"""
        if self.model is None:
            self.load_model()
        
        if not self.class_names:
            self.load_class_names()
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Kamera açılamadı: {camera_index}")
            return
        
        print("📹 Kamera başlatıldı. 'q' tuşuna basarak çıkın, 's' ile fotoğraf çekin.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame okunamadı")
                break
            
            # Frame'i göster
            display_frame = frame.copy()
            
            try:
                if continuous:
                    # Sürekli tahmin yap
                    predictions = self.predict_single_image(
                        frame, top_k=3, show_image=False
                    )
                    
                    # Tahminleri frame üzerine yaz
                    y_offset = 30
                    for pred in predictions[:3]:
                        text = f"{pred['class']}: {pred['percentage']:.1f}%"
                        cv2.putText(display_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                
                cv2.imshow('Fruit Predictor', display_frame)
                
            except Exception as e:
                cv2.putText(display_frame, f"Hata: {str(e)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Fruit Predictor', display_frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Fotoğraf çek ve tahmin yap
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Fotoğraf kaydedildi: {filename}")
                
                predictions = self.predict_single_image(filename, show_image=True)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_prediction_result(self, image_path, predictions, save_path=None):
        """Tahmin sonucunu görsel olarak göster"""
        # Orijinal görüntüyü yükle
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grafik oluştur
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Görüntüyü göster
        ax1.imshow(img)
        ax1.set_title(f'Görüntü: {os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Tahmin sonuçlarını bar chart olarak göster
        classes = [pred['class'] for pred in predictions]
        confidences = [pred['percentage'] for pred in predictions]
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        
        bars = ax2.barh(range(len(classes)), confidences, color=colors)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Güven (%)')
        ax2.set_title('Tahmin Sonuçları')
        ax2.grid(True, alpha=0.3)
        
        # En yüksek tahmini vurgula
        bars[0].set_color('gold')
        
        # Değerleri çubukların üzerine yaz
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Sonuç görseli kaydedildi: {save_path}")
        
        plt.show()
    
    def save_batch_results(self, batch_results):
        """Batch tahmin sonuçlarını kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config['paths']['predictions_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # JSON olarak kaydet
        json_path = os.path.join(results_dir, f'batch_predictions_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        # CSV olarak da kaydet
        csv_path = os.path.join(results_dir, f'batch_predictions_{timestamp}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Filename,Top1_Class,Top1_Confidence,Top2_Class,Top2_Confidence,Top3_Class,Top3_Confidence\n')
            
            for result in batch_results:
                if 'error' not in result:
                    filename = result['filename']
                    preds = result['predictions']
                    
                    row = [filename]
                    for i in range(3):
                        if i < len(preds):
                            row.extend([preds[i]['class'], f"{preds[i]['percentage']:.2f}"])
                        else:
                            row.extend(['', ''])
                    
                    f.write(','.join(map(str, row)) + '\n')
        
        print(f"✅ Batch sonuçları kaydedildi:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def create_prediction_gallery(self, batch_results, save_path=None):
        """Batch tahmin sonuçları için galeri oluştur"""
        valid_results = [r for r in batch_results if 'error' not in r]
        
        if not valid_results:
            print("❌ Geçerli tahmin sonucu bulunamadı!")
            return
        
        # Grid boyutunu hesapla
        n_images = min(len(valid_results), 20)  # Maksimum 20 görüntü
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            
            if i < n_images:
                result = valid_results[i]
                
                try:
                    # Görüntüyü yükle
                    img = cv2.imread(result['image_path'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    ax.imshow(img)
                    
                    # En iyi tahmin
                    top_pred = result['predictions'][0]
                    title = f"{result['filename']}\n{top_pred['class']}\n{top_pred['percentage']:.1f}%"
                    ax.set_title(title, fontsize=10)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Hata:\n{str(e)}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(result['filename'])
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Tahmin galerisi kaydedildi: {save_path}")
        
        plt.show()


def main():
    """Ana fonksiyon - komut satırı arayüzü"""
    parser = argparse.ArgumentParser(description='Fruits-360 Model Tahmin Aracı')
    parser.add_argument('model_path', help='Model dosya yolu (.h5)')
    parser.add_argument('--image', '-i', help='Tek görüntü dosya yolu')
    parser.add_argument('--batch', '-b', help='Görüntü klasörü yolu')
    parser.add_argument('--camera', '-c', action='store_true', help='Kamera modunu başlat')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='En iyi K tahmini göster (varsayılan: 5)')
    parser.add_argument('--class-names', help='Sınıf isimleri JSON dosyası')
    parser.add_argument('--no-show', action='store_true', help='Görüntüleri gösterme')
    
    args = parser.parse_args()
    
    # Predictor'ı başlat
    predictor = FruitPredictor(args.model_path)
    predictor.load_model()
    predictor.load_class_names(args.class_names)
    
    if args.camera:
        # Kamera modu
        predictor.predict_from_camera()
        
    elif args.image:
        # Tek görüntü modu
        if not os.path.exists(args.image):
            print(f"❌ Görüntü dosyası bulunamadı: {args.image}")
            return
        
        predictions = predictor.predict_single_image(
            args.image, 
            top_k=args.top_k, 
            show_image=not args.no_show
        )
        
    elif args.batch:
        # Batch modu
        if not os.path.exists(args.batch):
            print(f"❌ Klasör bulunamadı: {args.batch}")
            return
        
        # Görüntü dosyalarını bul
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for file in os.listdir(args.batch):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.batch, file))
        
        if not image_paths:
            print(f"❌ Klasörde görüntü dosyası bulunamadı: {args.batch}")
            return
        
        print(f"📁 {len(image_paths)} görüntü bulundu")
        
        # Batch tahmin
        batch_results = predictor.predict_batch_images(image_paths, top_k=args.top_k)
        
        # Galeri oluştur
        if not args.no_show:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gallery_path = os.path.join(
                predictor.config['paths']['predictions_dir'],
                f'prediction_gallery_{timestamp}.png'
            )
            predictor.create_prediction_gallery(batch_results, gallery_path)
        
    else:
        print("❌ Lütfen --image, --batch veya --camera seçeneklerinden birini kullanın")
        parser.print_help()


if __name__ == "__main__":
    main()