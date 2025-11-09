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

# Buat folder untuk menyimpan hasil evaluasi
os.makedirs("evaluation_results", exist_ok=True)

# Load class mapping
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())
num_classes = len(class_names)



# Konfigurasi model yang akan dievaluasi
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

# ============================================================================
# 1. EVALUASI AKURASI DAN METRIK KLASIFIKASI
# ============================================================================

def evaluate_classification_metrics(model_name, model_path, target_size):
    """Evaluasi metrik klasifikasi: accuracy, precision, recall, F1-score"""
    
    print(f"\n{'='*80}")
    print(f"EVALUASI: {model_name}")
    print(f"{'='*80}")
    print(f"Deskripsi: {models_config[model_name]['description']}")
    print(f"Input Size: {target_size}")
    
    # Load model
    print("\nMemuat model...")
    model = load_model(model_path)
    model.summary()
    
    # Hitung parameter model
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Prepare test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        "data/dataset_makanan_indonesia",
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Jumlah sampel test: {test_gen.samples}")
    
    # Evaluate
    print("\nMenghitung loss dan akurasi...")
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    
    # Get predictions
    print("\nMelakukan prediksi pada test set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Overall metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("HASIL EVALUASI")
    print(f"{'='*60}")
    print(f"Test Loss:           {loss:.4f}")
    print(f"Test Accuracy:       {accuracy*100:.2f}%")
    print(f"Weighted Precision:  {precision_avg:.4f}")
    print(f"Weighted Recall:     {recall_avg:.4f}")
    print(f"Weighted F1-Score:   {f1_avg:.4f}")
    print(f"{'='*60}")
    
    # Create detailed classification report
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
    
    # Create DataFrame untuk per-class metrics
    per_class_df = pd.DataFrame({
        'Kelas': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Sort by F1-Score
    per_class_df = per_class_df.sort_values('F1-Score', ascending=False)
    
    print("\nPERFORMA PER KELAS (Sorted by F1-Score):")
    print(per_class_df.to_string(index=False))
    
    # Save to CSV
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

# ============================================================================
# 2. CONFUSION MATRIX
# ============================================================================

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

# ============================================================================
# 3. CONFIDENCE SCORE ANALYSIS
# ============================================================================

def analyze_confidence(predictions, y_true, y_pred, model_name):
    """Analisis confidence score dari prediksi"""
    
    print(f"\nANALISIS CONFIDENCE SCORE - {model_name}")
    print("="*60)
    
    # Get confidence scores (max probability)
    confidences = np.max(predictions, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    print(f"Rata-rata confidence (benar):    {np.mean(correct_confidences):.4f}")
    print(f"Rata-rata confidence (salah):    {np.mean(incorrect_confidences):.4f}")
    print(f"Median confidence (benar):       {np.median(correct_confidences):.4f}")
    print(f"Median confidence (salah):       {np.median(incorrect_confidences):.4f}")
    print(f"Min confidence (benar):          {np.min(correct_confidences):.4f}")
    print(f"Max confidence (salah):          {np.max(incorrect_confidences):.4f}")
    
    # Plot confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
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
    
    # Box plot
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
    
    # Confidence threshold analysis
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

# ============================================================================
# 4. SPEED BENCHMARK
# ============================================================================

def benchmark_inference_speed(model_name, model_path, target_size, iterations=100):
    """Benchmark kecepatan inference model"""
    
    print(f"\nBENCHMARK KECEPATAN INFERENCE - {model_name}")
    print("="*60)
    
    model = load_model(model_path)
    
    # Create dummy input
    dummy_input = np.random.rand(1, *target_size, 3).astype(np.float32)
    
    # Warmup
    print("Warmup (10 iterations)...")
    for _ in range(10):
        model.predict(dummy_input, verbose=0)
    
    # Benchmark
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

# ============================================================================
# 5. TRAINING HISTORY VISUALIZATION
# ============================================================================

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
    
    # Check which history files exist
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
    
    # Plot individual model curves
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
            
            # Add final values as text
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
            
            # Validation Accuracy comparison
            if 'val_accuracy' in history:
                epochs = range(1, len(history['val_accuracy']) + 1)
                axes[0].plot(epochs, history['val_accuracy'], label=model_name, 
                           linewidth=2, color=color)
            
            # Validation Loss comparison
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

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================
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

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RINGKASAN KOMPARASI MODEL")
print("="*80)

# Create comparison DataFrame
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

# Save to CSV
comparison_path = "evaluation_results/model_comparison_summary.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"\nRingkasan komparasi disimpan ke: {comparison_path}")

# Visualisasi perbandingan model
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

# ============================================================================
# FINAL REPORT
# ============================================================================

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