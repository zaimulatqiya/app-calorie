import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
from itertools import combinations


os.makedirs("evaluation_results", exist_ok=True)


with open("models/class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())
num_classes = len(class_names)




models_config = {
    "MobileNetV2": {
        "path": "models/food_model_mobilenet.h5",
        "size": (128, 128),
        "description": "Transfer learning dengan MobileNetV2"
    },
    "EfficientNetB0": {
        "path": "models/food_model_efficientnet_finetuned.h5",
        "size": (224, 224),
        "description": "Fine-tuned EfficientNetB0"
    },
    "SimpleCNN": {
        "path": "models/food_model.h5",
        "size": (128, 128),
        "description": "Custom CNN sederhana"
    }
}

print("="*80)
print("EVALUASI KOMPREHENSIF MODEL KLASIFIKASI MAKANAN INDONESIA")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Jumlah Kelas: {num_classes}")
print(f"Kelas Makanan: {', '.join(class_names)}")

assert num_classes == 18, f"❌ ERROR: Expected 18 classes, got {num_classes}"
print("✅ Validasi: Jumlah kelas sesuai (18)")
print("="*80)



def evaluate_classification_metrics(model_name, model_path, target_size):
    """Evaluasi metrik klasifikasi: accuracy, precision, recall, F1-score"""
    
    print(f"\n{'='*80}")
    print(f"EVALUASI: {model_name}")
    print(f"{'='*80}")
    print(f"Deskripsi: {models_config[model_name]['description']}")
    print(f"Input Size: {target_size}")
    
    
    print("\nMemuat model...")
    model = load_model(model_path)
    model.summary()
    
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        "data/dataset_makanan_indonesia",
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Jumlah sampel test: {test_gen.samples}")
    
    
    print("\nMenghitung loss dan akurasi...")
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    
   
    print("\nMelakukan prediksi pada test set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
   
    print(f"\n{'='*60}")
    print("HASIL EVALUASI")
    print(f"{'='*60}")
    print(f"Test Loss:           {loss:.4f}")
    print(f"Test Accuracy:       {accuracy*100:.2f}%")
    print(f"Weighted Precision:  {precision_avg:.4f}")
    print(f"Weighted Recall:     {recall_avg:.4f}")
    print(f"Weighted F1-Score:   {f1_avg:.4f}")
    print(f"{'='*60}")
    
   
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT PER KELAS")
    print(f"{'='*60}")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Save classification report to file
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    
    per_class_df = pd.DataFrame({
        'Kelas': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    
    per_class_df = per_class_df.sort_values('F1-Score', ascending=False)
    
    print("\nPERFORMA PER KELAS (Sorted by F1-Score):")
    print(per_class_df.to_string(index=False))
    
    
    csv_path = f"evaluation_results/{model_name}_per_class_metrics.csv"
    per_class_df.to_csv(csv_path, index=False)
    print(f"\nMetrik per kelas disimpan ke: {csv_path}")
    
    # Visualisasi metrik per kelas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (metric, values) in enumerate([
        ('Precision', precision),
        ('Recall', recall),
        ('F1-Score', f1)
    ]):
        ax = axes[idx]
        y_pos = np.arange(len(class_names))
        colors = plt.cm.RdYlGn(values)
        
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_title(f'{metric} per Kelas', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', 
                   va='center', fontsize=7)
    
    plt.tight_layout()
    fig_path = f"evaluation_results/{model_name}_per_class_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Grafik metrik per kelas disimpan ke: {fig_path}")
    plt.close()
    
    return {
        'model': model_name,
        'loss': float(loss),
        'accuracy': float(accuracy),
        'precision': float(precision_avg),
        'recall': float(recall_avg),
        'f1_score': float(f1_avg),
        'total_params': int(total_params),
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': predictions,
        'per_class_metrics': per_class_df
    }



def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot dan simpan confusion matrix"""
    
    print(f"\nMembuat Confusion Matrix untuk {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix (raw counts)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}\n(Raw Counts)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=8)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f'Confusion Matrix - {model_name}\n(Normalized)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    cm_path = f"evaluation_results/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix disimpan ke: {cm_path}")
    plt.close()
    
    # Analisis kesalahan klasifikasi terbesar
    print(f"\nANALISIS KESALAHAN KLASIFIKASI ({model_name}):")
    print("="*60)
    
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'True': class_names[i],
                    'Predicted': class_names[j],
                    'Count': cm[i, j],
                    'Percentage': cm_normalized[i, j] * 100
                })
    
    if misclassifications:
        misclass_df = pd.DataFrame(misclassifications)
        misclass_df = misclass_df.sort_values('Count', ascending=False).head(10)
        print("Top 10 Kesalahan Klasifikasi:")
        print(misclass_df.to_string(index=False))
        
        # Save to CSV
        misclass_path = f"evaluation_results/{model_name}_misclassifications.csv"
        misclass_df.to_csv(misclass_path, index=False)
        print(f"\nKesalahan klasifikasi disimpan ke: {misclass_path}")
    else:
        print("Tidak ada kesalahan klasifikasi (Perfect prediction!)")



def analyze_confidence(predictions, y_true, y_pred, model_name):
    """Analisis confidence score dari prediksi"""
    
    print(f"\nANALISIS CONFIDENCE SCORE - {model_name}")
    print("="*60)
    
    
    confidences = np.max(predictions, axis=1)
    
    
    correct_mask = (y_true == y_pred)
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    print(f"Rata-rata confidence (benar):    {np.mean(correct_confidences):.4f}")
    print(f"Rata-rata confidence (salah):    {np.mean(incorrect_confidences):.4f}")
    print(f"Median confidence (benar):       {np.median(correct_confidences):.4f}")
    print(f"Median confidence (salah):       {np.median(incorrect_confidences):.4f}")
    print(f"Min confidence (benar):          {np.min(correct_confidences):.4f}")
    print(f"Max confidence (salah):          {np.max(incorrect_confidences):.4f}")
    
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    
    axes[0].hist(correct_confidences, bins=30, alpha=0.7, 
                label='Correct', color='green', edgecolor='black')
    axes[0].hist(incorrect_confidences, bins=30, alpha=0.7,
                label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Distribution of Confidence Scores\n{model_name}',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
   
    data_to_plot = [correct_confidences, incorrect_confidences]
    bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                         patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence Score', fontsize=11)
    axes[1].set_title(f'Confidence Score Comparison\n{model_name}',
                     fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    conf_path = f"evaluation_results/{model_name}_confidence_analysis.png"
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafik confidence analysis disimpan ke: {conf_path}")
    plt.close()
    
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nAKURASI BERDASARKAN THRESHOLD CONFIDENCE:")
    print("-"*60)
    
    for threshold in thresholds:
        mask = confidences >= threshold
        if np.sum(mask) > 0:
            acc_at_threshold = np.mean((y_true[mask] == y_pred[mask]))
            coverage = np.sum(mask) / len(confidences)
            print(f"Threshold >= {threshold:.1f}: "
                  f"Accuracy = {acc_at_threshold*100:.2f}%, "
                  f"Coverage = {coverage*100:.2f}%")
        else:
            print(f"Threshold >= {threshold:.1f}: No predictions")



def benchmark_inference_speed(model_name, model_path, target_size, iterations=100):
    """Benchmark kecepatan inference model"""
    
    print(f"\nBENCHMARK KECEPATAN INFERENCE - {model_name}")
    print("="*60)
    
    model = load_model(model_path)
    
    
    dummy_input = np.random.rand(1, *target_size, 3).astype(np.float32)
    
    
    print("Warmup (10 iterations)...")
    for _ in range(10):
        model.predict(dummy_input, verbose=0)
    
    
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        times.append(time.time() - start)
    
    times_ms = np.array(times) * 1000
    
    print(f"\nHASIL BENCHMARK:")
    print(f"  Mean:       {np.mean(times_ms):.2f} ms")
    print(f"  Median:     {np.median(times_ms):.2f} ms")
    print(f"  Std Dev:    {np.std(times_ms):.2f} ms")
    print(f"  Min:        {np.min(times_ms):.2f} ms")
    print(f"  Max:        {np.max(times_ms):.2f} ms")
    print(f"  P95:        {np.percentile(times_ms, 95):.2f} ms")
    print(f"  P99:        {np.percentile(times_ms, 99):.2f} ms")
    
    # Get model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nUkuran Model: {model_size_mb:.2f} MB")
    
    return {
        'model': model_name,
        'mean_ms': float(np.mean(times_ms)),
        'median_ms': float(np.median(times_ms)),
        'std_ms': float(np.std(times_ms)),
        'min_ms': float(np.min(times_ms)),
        'max_ms': float(np.max(times_ms)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'model_size_mb': float(model_size_mb),
        'times': times_ms
    }



def plot_training_curves():
    """Plot training history curves jika file history tersedia"""
    
    print(f"\n{'='*80}")
    print("VISUALISASI TRAINING CURVES")
    print(f"{'='*80}")
    
    history_files = {
        "SimpleCNN": "training_history_simplecnn.json",
        "MobileNetV2": "training_history_mobilenet.json",
        "EfficientNetB0": "training_history_efficientnet.json"
    }
    
    available_histories = {}
    
    
    for model_name, filename in history_files.items():
        filepath = os.path.join("models", filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    available_histories[model_name] = json.load(f)
                print(f"✅ History ditemukan untuk {model_name}")
            except Exception as e:
                print(f"❌ Error loading history {model_name}: {e}")
        else:
            print(f"⚠️  History tidak ditemukan: {filepath}")
    
    if not available_histories:
        print("\n⚠️  Tidak ada training history yang ditemukan.")
        print("Untuk mengaktifkan visualisasi ini, simpan history saat training:")
        print("   import json")
        print("   with open('models/training_history_[model].json', 'w') as f:")
        print("       json.dump(history.history, f)")
        return
    
    
    for model_name, history in available_histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy curve
        if 'accuracy' in history and 'val_accuracy' in history:
            epochs = range(1, len(history['accuracy']) + 1)
            axes[0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Accuracy', fontsize=11)
            axes[0].set_title(f'{model_name} - Accuracy vs Epoch', fontsize=12, fontweight='bold')
            axes[0].legend(loc='lower right')
            axes[0].grid(alpha=0.3)
            
            # Add final values as text
            final_train = history['accuracy'][-1]
            final_val = history['val_accuracy'][-1]
            axes[0].text(0.02, 0.98, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
                        transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Loss curve
        if 'loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            axes[1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
            axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Loss', fontsize=11)
            axes[1].set_title(f'{model_name} - Loss vs Epoch', fontsize=12, fontweight='bold')
            axes[1].legend(loc='upper right')
            axes[1].grid(alpha=0.3)
            
            
            final_train = history['loss'][-1]
            final_val = history['val_loss'][-1]
            axes[1].text(0.02, 0.98, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
                        transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        curve_path = f"evaluation_results/{model_name}_training_curves.png"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"Training curves disimpan ke: {curve_path}")
        plt.close()
    
    # Plot comparison of all models
    if len(available_histories) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'SimpleCNN': '#FF9800', 'MobileNetV2': '#4CAF50', 'EfficientNetB0': '#2196F3'}
        
        for model_name, history in available_histories.items():
            color = colors.get(model_name, 'gray')
            
            
            if 'val_accuracy' in history:
                epochs = range(1, len(history['val_accuracy']) + 1)
                axes[0].plot(epochs, history['val_accuracy'], label=model_name, 
                           linewidth=2, color=color)
            
            
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                axes[1].plot(epochs, history['val_loss'], label=model_name, 
                           linewidth=2, color=color)
        
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Validation Accuracy', fontsize=11)
        axes[0].set_title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Validation Loss', fontsize=11)
        axes[1].set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        comparison_path = "evaluation_results/training_curves_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafik perbandingan training curves disimpan ke: {comparison_path}")
        plt.close()
    
    print(f"{'='*80}\n")



def calculate_per_sample_accuracy(y_true, y_pred):
    """Calculate per-sample accuracy (1 if correct, 0 if incorrect)"""
    return (y_true == y_pred).astype(int)

def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    Perform McNemar's test to compare two models.
    Returns: statistic, p-value, contingency table
    """
    # Create contingency table
    # [correct_both, model1_correct_only]
    # [model2_correct_only, both_incorrect]
    correct_model1 = (y_true == y_pred_model1).astype(int)
    correct_model2 = (y_true == y_pred_model2).astype(int)
    
    correct_both = np.sum((correct_model1 == 1) & (correct_model2 == 1))
    model1_only = np.sum((correct_model1 == 1) & (correct_model2 == 0))
    model2_only = np.sum((correct_model1 == 0) & (correct_model2 == 1))
    both_wrong = np.sum((correct_model1 == 0) & (correct_model2 == 0))
    
    contingency_table = np.array([[correct_both, model1_only],
                                   [model2_only, both_wrong]])
    
    # McNemar's test statistic
    # Only uses the off-diagonal elements (b and c)
    b = model1_only
    c = model2_only
    
    if b + c == 0:
        # No discordant pairs
        return 0, 1.0, contingency_table
    
    # McNemar's test with continuity correction
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value, contingency_table

def paired_ttest_models(all_results):
    """
    Perform paired t-test between all pairs of models
    """
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    print("\nMelakukan uji statistik untuk membandingkan performa model...")
    print("Metode: Paired t-test dan McNemar's test")
    print(f"{'='*80}\n")
    
    if len(all_results) < 2:
        print("⚠️  Minimal 2 model diperlukan untuk statistical testing")
        return None
    
    
    y_true_ref = all_results[0]['y_true']
    for result in all_results[1:]:
        if not np.array_equal(result['y_true'], y_true_ref):
            print("⚠️  WARNING: Models have different test sets!")
            return None
    
    statistical_results = []
    
    
    model_pairs = list(combinations(range(len(all_results)), 2))
    
    print("="*80)
    print("1. PAIRED T-TEST (Accuracy Comparison)")
    print("="*80)
    print("Hipotesis H0: Tidak ada perbedaan signifikan antara dua model")
    print("Hipotesis H1: Ada perbedaan signifikan antara dua model")
    print("Significance level: α = 0.05\n")
    
    for i, j in model_pairs:
        model1 = all_results[i]
        model2 = all_results[j]
        
        
        acc1 = calculate_per_sample_accuracy(model1['y_true'], model1['y_pred'])
        acc2 = calculate_per_sample_accuracy(model2['y_true'], model2['y_pred'])
        
        
        t_statistic, p_value = ttest_rel(acc1, acc2)
        
        
        diff = acc1 - acc2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        n = len(diff)
        se_diff = std_diff / np.sqrt(n)
        
        
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_diff, scale=se_diff)
        
        
        is_significant = p_value < 0.05
        
        print(f"\n{'─'*80}")
        print(f"Perbandingan: {model1['model']} vs {model2['model']}")
        print(f"{'─'*80}")
        print(f"Accuracy {model1['model']}:     {model1['accuracy']*100:.2f}%")
        print(f"Accuracy {model2['model']}:     {model2['accuracy']*100:.2f}%")
        print(f"Mean Difference:           {mean_diff*100:.4f}%")
        print(f"95% Confidence Interval:   [{ci_95[0]*100:.4f}%, {ci_95[1]*100:.4f}%]")
        print(f"\nT-statistic:               {t_statistic:.4f}")
        print(f"P-value:                   {p_value:.6f}")
        print(f"Degrees of freedom:        {n-1}")
        
        if is_significant:
            if mean_diff > 0:
                better_model = model1['model']
            else:
                better_model = model2['model']
            print(f"\n✅ SIGNIFIKAN (p < 0.05)")
            print(f"   → {better_model} secara statistik LEBIH BAIK")
        else:
            print(f"\n❌ TIDAK SIGNIFIKAN (p ≥ 0.05)")
            print(f"   → Tidak ada perbedaan yang signifikan")
        
        statistical_results.append({
            'Model 1': model1['model'],
            'Model 2': model2['model'],
            'Accuracy 1 (%)': f"{model1['accuracy']*100:.2f}",
            'Accuracy 2 (%)': f"{model2['accuracy']*100:.2f}",
            'Mean Diff (%)': f"{mean_diff*100:.4f}",
            'T-statistic': f"{t_statistic:.4f}",
            'P-value': f"{p_value:.6f}",
            'CI 95% Lower': f"{ci_95[0]*100:.4f}%",
            'CI 95% Upper': f"{ci_95[1]*100:.4f}%",
            'Significant': 'Yes' if is_significant else 'No',
            'Better Model': better_model if is_significant else 'No difference'
        })
    
    print(f"\n{'='*80}")
    print("2. McNEMAR'S TEST (Classification Disagreement)")
    print(f"{'='*80}")
    print("Uji untuk membandingkan kesalahan prediksi antara dua model")
    print("Hipotesis H0: Kedua model memiliki error rate yang sama")
    print("Hipotesis H1: Model memiliki error rate yang berbeda\n")
    
    mcnemar_results = []
    
    for i, j in model_pairs:
        model1 = all_results[i]
        model2 = all_results[j]
        
        
        statistic, p_value, contingency = mcnemar_test(
            model1['y_true'], model1['y_pred'], model2['y_pred']
        )
        
        is_significant = p_value < 0.05
        
        print(f"\n{'─'*80}")
        print(f"Perbandingan: {model1['model']} vs {model2['model']}")
        print(f"{'─'*80}")
        print("Contingency Table:")
        print(f"  Both Correct:              {contingency[0,0]:>6}")
        print(f"  {model1['model']} Only:        {contingency[0,1]:>6}")
        print(f"  {model2['model']} Only:        {contingency[1,0]:>6}")
        print(f"  Both Incorrect:            {contingency[1,1]:>6}")
        print(f"\nMcNemar Statistic:         {statistic:.4f}")
        print(f"P-value:                   {p_value:.6f}")
        
        if is_significant:
            print(f"✅ SIGNIFIKAN (p < 0.05)")
            print(f"   → Model memiliki pola kesalahan yang berbeda secara signifikan")
        else:
            print(f"❌ TIDAK SIGNIFIKAN (p ≥ 0.05)")
            print(f"   → Model memiliki pola kesalahan yang serupa")
        
        mcnemar_results.append({
            'Model 1': model1['model'],
            'Model 2': model2['model'],
            'Both Correct': contingency[0,0],
            'Model 1 Only': contingency[0,1],
            'Model 2 Only': contingency[1,0],
            'Both Wrong': contingency[1,1],
            'McNemar Stat': f"{statistic:.4f}",
            'P-value': f"{p_value:.6f}",
            'Significant': 'Yes' if is_significant else 'No'
        })
    
    # Save statistical results
    if statistical_results:
        stats_df = pd.DataFrame(statistical_results)
        stats_path = "evaluation_results/paired_ttest_results.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\n{'='*80}")
        print(f"✅ Paired t-test results disimpan ke: {stats_path}")
        
        mcnemar_df = pd.DataFrame(mcnemar_results)
        mcnemar_path = "evaluation_results/mcnemar_test_results.csv"
        mcnemar_df.to_csv(mcnemar_path, index=False)
        print(f"✅ McNemar test results disimpan ke: {mcnemar_path}")
    
    
    if len(statistical_results) > 0:
        visualize_statistical_results(statistical_results, all_results)
    
    print(f"{'='*80}\n")
    
    return statistical_results

def visualize_statistical_results(statistical_results, all_results):
    """Create visualizations for statistical test results"""
    
    
    comparisons = [f"{r['Model 1']}\nvs\n{r['Model 2']}" for r in statistical_results]
    p_values = [float(r['P-value']) for r in statistical_results]
    mean_diffs = [float(r['Mean Diff (%)']) for r in statistical_results]
    is_significant = [r['Significant'] == 'Yes' for r in statistical_results]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    
    colors = ['green' if sig else 'red' for sig in is_significant]
    axes[0].bar(range(len(p_values)), p_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 (significance level)')
    axes[0].set_xlabel('Model Comparison', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('P-value', fontsize=11, fontweight='bold')
    axes[0].set_title('Paired T-Test P-values\n(Lower = More Significant Difference)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(comparisons)))
    axes[0].set_xticklabels(comparisons, fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    
    for i, (pv, sig) in enumerate(zip(p_values, is_significant)):
        label = f'p={pv:.4f}\n{"✅ Sig" if sig else "❌ Not Sig"}'
        axes[0].text(i, pv + 0.02, label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
   
    ci_lowers = [float(r['CI 95% Lower'].rstrip('%')) for r in statistical_results]
    ci_uppers = [float(r['CI 95% Upper'].rstrip('%')) for r in statistical_results]
    errors = [[md - cl for md, cl in zip(mean_diffs, ci_lowers)],
              [cu - md for md, cu in zip(mean_diffs, ci_uppers)]]
    
    axes[1].bar(range(len(mean_diffs)), mean_diffs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].errorbar(range(len(mean_diffs)), mean_diffs, yerr=errors, 
                    fmt='none', ecolor='black', capsize=10, capthick=2)
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Model Comparison', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Accuracy Difference (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean Accuracy Differences with 95% CI\n(Positive = Model 1 Better)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(comparisons)))
    axes[1].set_xticklabels(comparisons, fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
   
    for i, (md, sig) in enumerate(zip(mean_diffs, is_significant)):
        axes[1].text(i, md + (1 if md > 0 else -1), f'{md:.2f}%', 
                    ha='center', va='bottom' if md > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    stats_viz_path = "evaluation_results/statistical_significance_tests.png"
    plt.savefig(stats_viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Statistical test visualization disimpan ke: {stats_viz_path}")
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    
    table_data = []
    headers = ['Comparison', 'Acc 1', 'Acc 2', 'Diff', 'P-value', 'Significant', 'Winner']
    
    for r in statistical_results:
        row = [
            f"{r['Model 1']} vs {r['Model 2']}",
            r['Accuracy 1 (%)'] + '%',
            r['Accuracy 2 (%)'] + '%',
            r['Mean Diff (%)'] + '%',
            r['P-value'],
            '✅' if r['Significant'] == 'Yes' else '❌',
            r['Better Model']
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.1, 0.1, 0.1, 0.12, 0.12, 0.21])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if j == 5:  # Significant column
                if table_data[i-1][j] == '✅':
                    cell.set_facecolor('#C8E6C9')
                else:
                    cell.set_facecolor('#FFCDD2')
            else:
                cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
    
    plt.title('Statistical Significance Test Summary\nPaired T-Test Results', 
             fontsize=14, fontweight='bold', pad=20)
    
    table_path = "evaluation_results/statistical_test_summary_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"✅ Statistical test summary table disimpan ke: {table_path}")
    plt.close()


plot_training_curves()

all_results = []
speed_results = []

for model_name, config in models_config.items():
    try:
        # 1. Classification metrics
        result = evaluate_classification_metrics(
            model_name, config['path'], config['size']
        )
        all_results.append(result)
        
        # 2. Confusion matrix
        plot_confusion_matrix(
            result['y_true'], result['y_pred'], model_name
        )
        
        # 3. Confidence analysis
        analyze_confidence(
            result['predictions'], result['y_true'], 
            result['y_pred'], model_name
        )
        
        # 4. Speed benchmark
        speed_result = benchmark_inference_speed(
            model_name, config['path'], config['size']
        )
        speed_results.append(speed_result)
        
    except Exception as e:
        print(f"\nERROR evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()




if len(all_results) >= 2:
    statistical_test_results = paired_ttest_models(all_results)
else:
    print("\n⚠️  Statistical testing requires at least 2 models")
    statistical_test_results = None



print("\n" + "="*80)
print("RINGKASAN KOMPARASI MODEL")
print("="*80)


comparison_data = []
for result, speed in zip(all_results, speed_results):
    comparison_data.append({
        'Model': result['model'],
        'Accuracy (%)': f"{result['accuracy']*100:.2f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1-Score': f"{result['f1_score']:.4f}",
        'Loss': f"{result['loss']:.4f}",
        'Parameters': f"{result['total_params']:,}",
        'Inference (ms)': f"{speed['mean_ms']:.2f}",
        'Model Size (MB)': f"{speed['model_size_mb']:.2f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))


comparison_path = "evaluation_results/model_comparison_summary.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"\nRingkasan komparasi disimpan ke: {comparison_path}")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

model_names_list = [r['model'] for r in all_results]
accuracies = [r['accuracy']*100 for r in all_results]
f1_scores = [r['f1_score'] for r in all_results]
inference_times = [s['mean_ms'] for s in speed_results]
model_sizes = [s['model_size_mb'] for s in speed_results]

# Accuracy comparison
axes[0, 0].bar(model_names_list, accuracies, color=['#4CAF50', '#2196F3', '#FF9800'])
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim([0, 100])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')

# F1-Score comparison
axes[0, 1].bar(model_names_list, f1_scores, color=['#4CAF50', '#2196F3', '#FF9800'])
axes[0, 1].set_ylabel('F1-Score', fontsize=11)
axes[0, 1].set_title('Model F1-Score Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_scores):
    axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Inference speed comparison
axes[1, 0].bar(model_names_list, inference_times, color=['#4CAF50', '#2196F3', '#FF9800'])
axes[1, 0].set_ylabel('Inference Time (ms)', fontsize=11)
axes[1, 0].set_title('Model Inference Speed Comparison', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(inference_times):
    axes[1, 0].text(i, v + max(inference_times)*0.02, f'{v:.2f}ms', 
                   ha='center', fontweight='bold')

# Model size comparison
axes[1, 1].bar(model_names_list, model_sizes, color=['#4CAF50', '#2196F3', '#FF9800'])
axes[1, 1].set_ylabel('Model Size (MB)', fontsize=11)
axes[1, 1].set_title('Model Size Comparison', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(model_sizes):
    axes[1, 1].text(i, v + max(model_sizes)*0.02, f'{v:.2f}MB', 
                   ha='center', fontweight='bold')

plt.tight_layout()
comparison_fig_path = "evaluation_results/model_comparison_charts.png"
plt.savefig(comparison_fig_path, dpi=300, bbox_inches='tight')
print(f"Grafik perbandingan model disimpan ke: {comparison_fig_path}")
plt.close()

# Speed distribution comparison
plt.figure(figsize=(12, 6))
for i, (model_name, speed) in enumerate(zip(model_names_list, speed_results)):
    plt.hist(speed['times'], bins=30, alpha=0.6, label=model_name, 
            color=['#4CAF50', '#2196F3', '#FF9800'][i])
plt.xlabel('Inference Time (ms)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Inference Times', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
speed_dist_path = "evaluation_results/inference_time_distribution.png"
plt.savefig(speed_dist_path, dpi=300, bbox_inches='tight')
print(f"Distribusi waktu inference disimpan ke: {speed_dist_path}")
plt.close()



print("\n" + "="*80)
print("KESIMPULAN DAN REKOMENDASI")
print("="*80)

best_accuracy_idx = np.argmax([r['accuracy'] for r in all_results])
best_speed_idx = np.argmin([s['mean_ms'] for s in speed_results])
best_size_idx = np.argmin([s['model_size_mb'] for s in speed_results])

print(f"\nModel dengan AKURASI TERTINGGI:")
print(f"  {all_results[best_accuracy_idx]['model']}: "
      f"{all_results[best_accuracy_idx]['accuracy']*100:.2f}%")

print(f"\nModel TERCEPAT (inference):")
print(f"  {speed_results[best_speed_idx]['model']}: "
      f"{speed_results[best_speed_idx]['mean_ms']:.2f} ms")

print(f"\nModel dengan UKURAN TERKECIL:")
print(f"  {speed_results[best_size_idx]['model']}: "
      f"{speed_results[best_size_idx]['model_size_mb']:.2f} MB")

print("\nREKOMENDASI:")
print("-" * 80)
print("1. Untuk AKURASI MAKSIMAL: Gunakan model dengan accuracy tertinggi")
print("2. Untuk REAL-TIME APPLICATION: Gunakan model tercepat")
print("3. Untuk MOBILE/EDGE DEPLOYMENT: Gunakan model dengan ukuran terkecil")
print("4. TRADE-OFF: Pertimbangkan balance antara akurasi dan kecepatan")

print("\n" + "="*80)
print("EVALUASI SELESAI!")
print("="*80)
print(f"Semua hasil evaluasi disimpan di folder: evaluation_results/")
print("="*80)