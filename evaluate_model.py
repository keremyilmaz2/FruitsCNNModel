"""
Model Değerlendirme Modülü
Eğitilmiş modellerin detaylı değerlendirmesi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
import cv2
import json
from datetime import datetime
import pandas as pd

import config
import utils
import data_preprocessing

class ModelEvaluator:
    def __init__(self, model_path, config_dict=None):
        self.model_path = model_path
        self.config = config_dict or config.get_config()
        self.model = None
        self.class_names = []
        self.data_processor = None
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        
        print(f"Model yükleniyor: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        
        # Model bilgilerini göster
        print("Model başarıyla yüklendi!")
        print(f"Model boyutu: {utils.calculate_model_size(self.model_path)}")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        
        return self.model
    
    def load_test_data(self, method='generator'):
        """Test verilerini yükle"""
        print("Test verileri yükleniyor...")
        
        self.data_processor = data_preprocessing.create_data_processor(self.config)
        
        if method == 'generator':
            # ImageDataGenerator kullan
            _, _, test_generator = self.data_processor.create_data_generators(
                train_path=self.config['data']['train_path'],
                test_path=self.config['data']['test_path']
            )
            
            self.class_names = list(test_generator.class_indices.keys())
            return None, None, test_generator
            
        else:
            # Önişlenmiş veriyi yükle
            try:
                (_, _), (_, _), (X_test, y_test) = self.data_processor.load_preprocessed_data()
                self.class_names = self.data_processor.class_names
                return X_test, y_test, None
            except:
                # Veriyi sıfırdan yükle
                print("Önişlenmiş veri bulunamadı, test verisi yükleniyor...")
                X_test, y_test = self.data_processor.load_data_from_directory(
                    self.config['data']['test_path'], 'test'
                )
                _, y_test = self.data_processor.encode_labels(y_test)
                self.class_names = self.data_processor.class_names
                return X_test, y_test, None
    
    def evaluate_model(self, X_test=None, y_test=None, test_generator=None):
        """Modeli değerlendir"""
        if self.model is None:
            self.load_model()
        
        print("Model değerlendirmesi başlıyor...")
        
        if test_generator is not None:
            # Generator ile değerlendirme
            test_generator.reset()
            test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
            
            # Tahminler
            test_generator.reset()
            y_pred = self.model.predict(test_generator, verbose=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = test_generator.classes
            
        else:
            # Array ile değerlendirme
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
            
            # Tahminler
            y_pred = self.model.predict(X_test, verbose=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
        
        # Sonuçları kaydet
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'y_true': y_true_classes.tolist(),
            'y_pred': y_pred_classes.tolist(),
            'y_pred_proba': y_pred.tolist(),
            'class_names': self.class_names,
            'evaluation_date': datetime.now().isoformat()
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def detailed_classification_report(self, results, save_path=None):
        """Detaylı sınıflandırma raporu"""
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # DataFrame'e çevir
        df_report = pd.DataFrame(report).transpose()
        
        print("Detaylı Sınıflandırma Raporu:")
        print("=" * 60)
        print(df_report.round(4))
        
        # En iyi ve en kötü performans gösteren sınıflar
        class_f1_scores = {name: report[name]['f1-score'] 
                          for name in self.class_names}
        
        best_classes = sorted(class_f1_scores.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        worst_classes = sorted(class_f1_scores.items(), 
                              key=lambda x: x[1])[:5]
        
        print(f"\n🏆 En İyi 5 Sınıf (F1-Score):")
        for name, score in best_classes:
            print(f"  {name}: {score:.4f}")
        
        print(f"\n⚠️ En Kötü 5 Sınıf (F1-Score):")
        for name, score in worst_classes:
            print(f"  {name}: {score:.4f}")
        
        # CSV olarak kaydet
        if save_path:
            df_report.to_csv(save_path.replace('.txt', '.csv'))
            
            # Text olarak da kaydet
            with open(save_path, 'w') as f:
                f.write("Detaylı Sınıflandırma Raporu\n")
                f.write("=" * 60 + "\n")
                f.write(str(df_report.round(4)))
                f.write(f"\n\nEn İyi 5 Sınıf (F1-Score):\n")
                for name, score in best_classes:
                    f.write(f"  {name}: {score:.4f}\n")
                f.write(f"\nEn Kötü 5 Sınıf (F1-Score):\n")
                for name, score in worst_classes:
                    f.write(f"  {name}: {score:.4f}\n")
            
            print(f"Rapor kaydedildi: {save_path}")
        
        return df_report
    
    def plot_confusion_matrix(self, results, save_path=None, figsize=(20, 16)):
        """Gelişmiş confusion matrix"""
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize edilmiş confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Ham confusion matrix
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalize edilmiş confusion matrix
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix kaydedildi: {save_path}")
        
        plt.show()
        
        # En çok karıştırılan sınıfları bul
        self.find_most_confused_classes(cm, save_path)
    
    def find_most_confused_classes(self, cm, save_path=None):
        """En çok karıştırılan sınıf çiftlerini bul"""
        confused_pairs = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / cm[i].sum() * 100
                    })
        
        # En çok karıştırılan çiftleri sırala
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"\n🤔 En Çok Karıştırılan 10 Sınıf Çifti:")
        print("-" * 80)
        print(f"{'Gerçek Sınıf':<20} {'Tahmin Edilen':<20} {'Sayı':<8} {'Yüzde':<8}")
        print("-" * 80)
        
        for pair in confused_pairs[:10]:
            print(f"{pair['true_class']:<20} {pair['predicted_class']:<20} "
                 f"{pair['count']:<8} {pair['percentage']:<8.2f}%")
        
        if save_path:
            confusion_path = save_path.replace('.png', '_confused_pairs.txt')
            with open(confusion_path, 'w') as f:
                f.write("En Çok Karıştırılan Sınıf Çiftleri\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Gerçek Sınıf':<20} {'Tahmin Edilen':<20} {'Sayı':<8} {'Yüzde':<8}\n")
                f.write("-" * 80 + "\n")
                
                for pair in confused_pairs[:20]:  # İlk 20'yi kaydet
                    f.write(f"{pair['true_class']:<20} {pair['predicted_class']:<20} "
                           f"{pair['count']:<8} {pair['percentage']:<8.2f}%\n")
            
            print(f"Karıştırılan çiftler kaydedildi: {confusion_path}")
    
    def plot_class_performance(self, results, save_path=None):
        """Sınıf bazında performans grafiği"""
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        # Her sınıf için metrikler
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # DataFrame oluştur
        performance_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # F1-Score'a göre sırala
        performance_df = performance_df.sort_values('F1-Score', ascending=True)
        
        # Grafik oluştur
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Precision
        axes[0, 0].barh(range(len(performance_df)), performance_df['Precision'])
        axes[0, 0].set_yticks(range(len(performance_df)))
        axes[0, 0].set_yticklabels(performance_df['Class'])
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 1].barh(range(len(performance_df)), performance_df['Recall'])
        axes[0, 1].set_yticks(range(len(performance_df)))
        axes[0, 1].set_yticklabels(performance_df['Class'])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[1, 0].barh(range(len(performance_df)), performance_df['F1-Score'])
        axes[1, 0].set_yticks(range(len(performance_df)))
        axes[1, 0].set_yticklabels(performance_df['Class'])
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_title('F1-Score by Class')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Support
        axes[1, 1].barh(range(len(performance_df)), performance_df['Support'])
        axes[1, 1].set_yticks(range(len(performance_df)))
        axes[1, 1].set_yticklabels(performance_df['Class'])
        axes[1, 1].set_xlabel('Support (Number of Samples)')
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sınıf performans grafiği kaydedildi: {save_path}")
        
        plt.show()
        
        return performance_df
    
    def analyze_prediction_confidence(self, results, save_path=None):
        """Tahmin güven analizi"""
        y_pred_proba = np.array(results['y_pred_proba'])
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        # Maksimum güven skorları
        max_confidences = np.max(y_pred_proba, axis=1)
        
        # Doğru ve yanlış tahminler
        correct_predictions = (y_true == y_pred)
        
        correct_confidences = max_confidences[correct_predictions]
        incorrect_confidences = max_confidences[~correct_predictions]
        
        # Güven dağılımı grafiği
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(correct_confidences, bins=50, alpha=0.7, label='Doğru Tahminler', color='green')
        plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Yanlış Tahminler', color='red')
        plt.xlabel('Güven Skoru')
        plt.ylabel('Frekans')
        plt.title('Tahmin Güven Dağılımı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Güven eşiği vs accuracy
        thresholds = np.arange(0.1, 1.0, 0.05)
        accuracies = []
        sample_counts = []
        
        for threshold in thresholds:
            high_confidence_mask = max_confidences >= threshold
            if np.sum(high_confidence_mask) > 0:
                accuracy = accuracy_score(
                    y_true[high_confidence_mask], 
                    y_pred[high_confidence_mask]
                )
                accuracies.append(accuracy)
                sample_counts.append(np.sum(high_confidence_mask))
            else:
                accuracies.append(0)
                sample_counts.append(0)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, accuracies, 'b-', label='Accuracy')
        plt.xlabel('Güven Eşiği')
        plt.ylabel('Accuracy')
        plt.title('Güven Eşiği vs Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, sample_counts, 'r-', label='Örnek Sayısı')
        plt.xlabel('Güven Eşiği')
        plt.ylabel('Örnek Sayısı')
        plt.title('Güven Eşiği vs Örnek Sayısı')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # İstatistikler
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Ortalama Güven (Doğru): {np.mean(correct_confidences):.3f}
        Ortalama Güven (Yanlış): {np.mean(incorrect_confidences):.3f}
        
        Medyan Güven (Doğru): {np.median(correct_confidences):.3f}
        Medyan Güven (Yanlış): {np.median(incorrect_confidences):.3f}
        
        Toplam Doğru: {len(correct_confidences)}
        Toplam Yanlış: {len(incorrect_confidences)}
        
        Yüksek Güvenli (>0.9) Doğru: {np.sum(correct_confidences > 0.9)}
        Yüksek Güvenli (>0.9) Yanlış: {np.sum(incorrect_confidences > 0.9)}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Güven analizi kaydedildi: {save_path}")
        
        plt.show()
        
        print(f"\n📊 Güven Analizi:")
        print(f"- Ortalama güven (doğru tahminler): {np.mean(correct_confidences):.3f}")
        print(f"- Ortalama güven (yanlış tahminler): {np.mean(incorrect_confidences):.3f}")
        print(f"- Yüksek güvenli (>0.9) doğru tahminler: {np.sum(correct_confidences > 0.9)}")
        print(f"- Yüksek güvenli (>0.9) yanlış tahminler: {np.sum(incorrect_confidences > 0.9)}")
    
    def full_evaluation_report(self, method='generator', save_dir=None):
        """Tam değerlendirme raporu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_dir is None:
            save_dir = self.config['paths']['results_dir']
        
        print("🔍 Tam model değerlendirmesi başlıyor...")
        print("=" * 60)
        
        # Modeli yükle
        self.load_model()
        
        # Test verilerini yükle
        X_test, y_test, test_generator = self.load_test_data(method)
        
        # Model değerlendirmesi
        results = self.evaluate_model(X_test, y_test, test_generator)
        
        # Sonuçları kaydet
        results_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Sonuçlar kaydedildi: {results_path}")
        
        # Detaylı classification report
        report_path = os.path.join(save_dir, f'classification_report_{timestamp}.txt')
        df_report = self.detailed_classification_report(results, report_path)
        
        # Confusion matrix
        cm_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
        self.plot_confusion_matrix(results, cm_path)
        
        # Sınıf performans grafiği
        perf_path = os.path.join(save_dir, f'class_performance_{timestamp}.png')
        performance_df = self.plot_class_performance(results, perf_path)
        
        # Güven analizi
        conf_path = os.path.join(save_dir, f'confidence_analysis_{timestamp}.png')
        self.analyze_prediction_confidence(results, conf_path)
        
        print(f"\n🎉 Tam değerlendirme tamamlandı!")
        print(f"📁 Sonuçlar klasörü: {save_dir}")
        
        return results, df_report, performance_df


def compare_models(model_paths, config_dict=None, method='generator'):
    """Birden fazla modeli karşılaştır"""
    print("🔄 Model karşılaştırması başlıyor...")
    
    if config_dict is None:
        config_dict = config.get_config()
    
    results_comparison = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\n📊 Model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        print("-" * 50)
        
        try:
            evaluator = ModelEvaluator(model_path, config_dict)
            evaluator.load_model()
            
            # Test verilerini yükle (sadece ilk modelde)
            if i == 0:
                X_test, y_test, test_generator = evaluator.load_test_data(method)
            else:
                # Aynı test verisini kullan
                evaluator.class_names = results_comparison[list(results_comparison.keys())[0]]['class_names']
                evaluator.data_processor = data_preprocessing.create_data_processor(config_dict)
                evaluator.data_processor.class_names = evaluator.class_names
            
            # Değerlendirme
            results = evaluator.evaluate_model(X_test, y_test, test_generator)
            
            model_name = os.path.basename(model_path).replace('.h5', '')
            results_comparison[model_name] = results
            
        except Exception as e:
            print(f"❌ Model değerlendirme hatası: {e}")
            continue
    
    # Karşılaştırma tablosu
    if len(results_comparison) > 1:
        create_comparison_table(results_comparison, config_dict['paths']['results_dir'])
    
    return results_comparison


def create_comparison_table(results_comparison, save_dir):
    """Model karşılaştırma tablosu oluştur"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Temel metrikler
    comparison_data = []
    for model_name, results in results_comparison.items():
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        # Metrikler hesapla
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        comparison_data.append({
            'Model': model_name,
            'Test Loss': results['test_loss'],
            'Test Accuracy': results['test_accuracy'],
            'Weighted Precision': precision,
            'Weighted Recall': recall,
            'Weighted F1-Score': f1
        })
    
    # DataFrame oluştur
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.round(4)
    
    print("\n📊 Model Karşılaştırma Tablosu:")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    
    # En iyi model
    best_model_idx = df_comparison['Test Accuracy'].idxmax()
    best_model = df_comparison.iloc[best_model_idx]['Model']
    best_accuracy = df_comparison.iloc[best_model_idx]['Test Accuracy']
    
    print(f"\n🏆 En iyi model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    # Kaydet
    csv_path = os.path.join(save_dir, f'model_comparison_{timestamp}.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"✅ Karşılaştırma tablosu kaydedildi: {csv_path}")
    
    # Grafik oluştur
    plot_comparison_chart(df_comparison, save_dir, timestamp)
    
    return df_comparison


def plot_comparison_chart(df_comparison, save_dir, timestamp):
    """Karşılaştırma grafiği"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Test Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        bars = ax.bar(df_comparison['Model'], df_comparison[metric])
        ax.set_title(f'{metric} Karşılaştırması')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # En yüksek değeri vurgula
        max_idx = df_comparison[metric].idxmax()
        bars[max_idx].set_color('gold')
        
        # Değerleri çubukların üzerine yaz
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    chart_path = os.path.join(save_dir, f'comparison_chart_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✅ Karşılaştırma grafiği kaydedildi: {chart_path}")
    
    plt.show()


# evaluate_model.py dosyasının en sonundaki if __name__ == "__main__": kısmını değiştirin

if __name__ == "__main__":
    import sys
    
    # ⭐ SPYDER İÇİN MANUAL MODEL TESTI:
    config_dict = config.get_config()
    
    # ESKİ MODEL TEST ET:
    print("🔍 ESKİ MODEL TEST EDİLİYOR...")
    model_path_old = 'models/fruits_resnet50_optimal_20250801_010023.h5'
    
    if os.path.exists(model_path_old):
        evaluator_old = ModelEvaluator(model_path_old, config_dict)
        results_old, _, _ = evaluator_old.full_evaluation_report('generator')
        
        print(f"\n📊 ESKİ MODEL SONUÇLARI:")
        print(f"Test Accuracy: {results_old['test_accuracy']:.4f}")
        print(f"Test Loss: {results_old['test_loss']:.4f}")
        
        # YENİ MODEL TEST ET:
        print("\n" + "="*60)
        print("🔍 YENİ MODEL TEST EDİLİYOR...")
        model_path_new = 'models/fruits_resnet50_finetune_20250801_095746.h5'
        
        if os.path.exists(model_path_new):
            evaluator_new = ModelEvaluator(model_path_new, config_dict)
            results_new, _, _ = evaluator_new.full_evaluation_report('generator')
            
            print(f"\n📊 YENİ MODEL SONUÇLARI:")
            print(f"Test Accuracy: {results_new['test_accuracy']:.4f}")
            print(f"Test Loss: {results_new['test_loss']:.4f}")
            
            # KARŞILAŞTIRMA:
            print("\n" + "="*60)
            print("🏆 MODEL KARŞILAŞTIRMASI:")
            print("="*60)
            print(f"ESKİ MODEL:  {results_old['test_accuracy']:.4f}")
            print(f"YENİ MODEL:  {results_new['test_accuracy']:.4f}")
            improvement = results_new['test_accuracy'] - results_old['test_accuracy']
            print(f"İYİLEŞME:    {improvement:+.4f} ({improvement*100:+.2f}%)")
            
            if improvement > 0:
                print("🎉 FINE-TUNING BAŞARILI!")
            else:
                print("⚠️ Fine-tuning'de sorun var")
        else:
            print(f"❌ Yeni model bulunamadı: {model_path_new}")
    else:
        print(f"❌ Eski model bulunamadı: {model_path_old}")