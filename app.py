from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary untuk menyimpan model yang telah dimuat
loaded_models = {}

# Load mapping kelas
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# Database kalori (per 100g)
# Default portion standar adalah 150g
DEFAULT_PORTION = 150

calorie_db = {
    "ayam_goreng": 275.0,
    "ayam_pop": 200.0,
    "bakso": 202.0,
    "bebek_betutu": 250.0,
    "dendeng_batokok": 300.0,
    "gado_gado": 137.0,
    "gudeg": 165.0,
    "gulai_ikan": 106.0,
    "gulai_tambusu": 180.0,
    "gulai_tunjang": 122.0,
    "nasi_goreng": 267.0,
    "pempek": 162.0,
    "rawon": 338.0,
    "rendang": 193.0,
    "sate": 200.0,
    "soto": 150.0,
    "telur_balado": 142.0,
    "telur_dadar": 154.0
}

# Mapping model paths
model_paths = {
    "mobilenet": "models/food_model_mobilenet.h5",
    "efficientnet": "models/food_model_efficientnet_finetuned.h5",
    "simplecnn": "models/food_model.h5"
}

# Mapping ukuran input tiap model
input_sizes = {
    "mobilenet": (128, 128),
    "efficientnet": (224, 224),
    "simplecnn": (128, 128)
}

# File untuk simpan riwayat
HISTORY_FILE = "history.json"

# Load history dari file jika ada
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_model_by_choice(model_choice):
    """Load model berdasarkan pilihan user dengan caching"""
    if model_choice not in loaded_models:
        model_path = model_paths.get(model_choice)
        if model_path and os.path.exists(model_path):
            loaded_models[model_choice] = load_model(model_path)
            print(f"✅ Model {model_choice} loaded from {model_path}")
        else:
            default_path = model_paths["mobilenet"]
            if os.path.exists(default_path):
                loaded_models[model_choice] = load_model(default_path)
                print(f"⚠️ Model {model_choice} not found, using default MobileNet model")
            else:
                raise FileNotFoundError(f"Model {model_choice} tidak ditemukan")
    return loaded_models[model_choice]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_food():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        model_choice = request.form.get("model_choice", "mobilenet")

        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        if model_choice not in model_paths:
            model_choice = "mobilenet"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        model = load_model_by_choice(model_choice)

        # Preprocessing sesuai model
        target_size = input_sizes.get(model_choice, (128, 128))
        img = image.load_img(filepath, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        food_name = labels[class_idx]
        confidence = round(float(np.max(preds[0]) * 100), 2)

        # Get kalori per 100g (langsung nilai float, bukan dictionary)
        per100g = calorie_db.get(food_name)
        
        if per100g is not None:
            # Gunakan DEFAULT_PORTION
            portion = DEFAULT_PORTION
            calories = round(per100g * portion / 100)
        else:
            per100g = None
            portion = None
            calories = "Tidak tersedia"

        image_url = filepath

        result = {
            "food_name": food_name.replace('_', ' ').title(),
            "calories": calories,
            "confidence": confidence,
            "model_used": model_choice,
            "portion": portion,
            "per100g": per100g,
            "image_url": image_url,
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        }

        # Simpan ke riwayat
        history.append(result)
        save_history()

        return jsonify(result)

    except Exception as e:
        print(f"❌ Error in analyze_food: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'})

@app.route('/recalculate', methods=['POST'])
def recalculate_calories():
    """Endpoint untuk menghitung ulang kalori dengan porsi baru"""
    try:
        data = request.get_json()
        food_name = data.get('food_name', '').lower().replace(' ', '_')
        new_portion = float(data.get('portion', 0))
        
        if new_portion <= 0:
            return jsonify({'error': 'Porsi harus lebih dari 0 gram'})
        
        if new_portion > 10000:
            return jsonify({'error': 'Porsi terlalu besar (maksimal 10kg)'})
        
        per100g = calorie_db.get(food_name)
        if per100g:
            new_calories = round(per100g * new_portion / 100)
            
            return jsonify({
                'success': True,
                'calories': new_calories,
                'portion': new_portion,
                'per100g': per100g
            })
        else:
            return jsonify({'error': 'Informasi kalori tidak tersedia'})
            
    except ValueError:
        return jsonify({'error': 'Format porsi tidak valid'})
    except Exception as e:
        print(f"❌ Error in recalculate_calories: {str(e)}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'})

@app.route('/history')
def get_history():
    return jsonify(history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        global history
        
        # Hapus semua file gambar di folder uploads
        for item in history:
            if item.get('image_url'):
                try:
                    if os.path.exists(item['image_url']):
                        os.remove(item['image_url'])
                        print(f"✅ Deleted image: {item['image_url']}")
                except Exception as e:
                    print(f"⚠️ Could not delete image: {e}")
        
        # Clear history
        history = []
        save_history()
        
        return jsonify({"message": "History cleared successfully", "success": True})
    except Exception as e:
        print(f"❌ Error in clear_history: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    try:
        index = int(request.form.get("index"))
        global history
        
        if 0 <= index < len(history):
            # Hapus file gambar jika ada
            item = history[index]
            if item.get('image_url'):
                try:
                    if os.path.exists(item['image_url']):
                        os.remove(item['image_url'])
                        print(f"✅ Deleted image: {item['image_url']}")
                except Exception as e:
                    print(f"⚠️ Could not delete image: {e}")
            
            # Hapus item dari history
            history.pop(index)
            save_history()
            
            return jsonify({"message": "Item deleted successfully", "success": True})
        else:
            return jsonify({"error": "Index tidak valid", "success": False}), 400
    except ValueError:
        return jsonify({"error": "Index harus berupa angka", "success": False}), 400
    except Exception as e:
        print(f"❌ Error in delete_history_item: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'loaded_models': list(loaded_models.keys()),
        'available_models': list(model_paths.keys())
    })

if __name__ == '__main__':
    try:
        load_model_by_choice("mobilenet")
        print("✅ Default model pre-loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load default model: {e}")
    
    app.run(debug=True)