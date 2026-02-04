from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import deepmimo as dm
import os

app = Flask(__name__)

# --- CONFIG ---
SCENARIO = "city_16_sanfrancisco_3p5"
CACHE_FILE = "channels_cache.npz"
THRESHOLD = 1e-12 # Minimum Power Threshold

# 1. Load Data
dataset = dm.load(SCENARIO)
if os.path.exists(CACHE_FILE):
    cache = np.load(CACHE_FILE)
    for i in range(len(dataset)): dataset[i].channel = cache[f'bs_{i}']
else:
    ch_params = dm.ChannelParameters()
    ch_params.bs_antenna.shape = [8, 1] 
    dataset.compute_channels(ch_params)
    np.savez_compressed(CACHE_FILE, **{f'bs_{i}': dataset[i].channel for i in range(len(dataset))})

# 2. Load Models
models = [tf.keras.models.load_model(f'./models/san_fran_BS_{i+1}_model.keras') for i in range(3)]

# 3. Codebook
def get_codebook(n_ant=8, n_beams=16):
    cb = np.zeros((n_ant, n_beams), dtype=complex)
    for q in range(n_beams):
        cb[:, q] = np.exp(-1j * 2 * np.pi * q * np.arange(n_ant) / n_beams) / np.sqrt(n_ant)
    return cb
CODEBOOK = get_codebook()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    norm_x, norm_y = data['x'], data['y']
    
    # Map click to meters
    user_coords = dataset[0].rx_pos
    min_x, max_x = np.min(user_coords[:, 0]), np.max(user_coords[:, 0])
    max_y, min_y = np.min(user_coords[:, 1]), np.max(user_coords[:, 1])
    target_x = min_x + (norm_x * (max_x - min_x))
    target_y = min_y + (norm_y * (max_y - min_y))

    # Snap to Grid
    u_idx = np.argmin(np.linalg.norm(user_coords[:, :2] - [target_x, target_y], axis=1))
    
    results = []
    powers = []
    
    for i in range(3):
        h = dataset[i].channel[u_idx].squeeze()
        h_power = np.linalg.norm(h)**2
        
        # Inference
        X_in = np.stack([np.real(h), np.imag(h)], axis=-1)
        X_in = (X_in / (np.max(np.abs(X_in)) + 1e-9))[np.newaxis, ..., np.newaxis]
        
        # Corrected Mapping: BS1=Model[0], BS2=Model[1], BS3=Model[2]
        # But Dataset[0]=BS3, Dataset[2]=BS1
        label = 3 - i
        pred_idx = np.argmax(models[label-1].predict(X_in, verbose=0))
        w = CODEBOOK[:, pred_idx]
        
        # Calculate Power & Gain
        p_val = np.abs(np.vdot(h, w))**2
        powers.append(p_val)
        
        gain_db = 10 * np.log10(p_val / (h_power + 1e-12) + 1e-12)

        results.append({
            'bs': label,
            'beam': int(pred_idx) if p_val > THRESHOLD else "N/A",
            'gain': round(float(gain_db), 2) if p_val > THRESHOLD else -100,
            'status': "Connected" if p_val > THRESHOLD else "Shadowed",
            'weights': [f"{np.abs(v):.2f}∠{np.angle(v, deg=True):.0f}°" for v in w] if p_val > THRESHOLD else []
        })

    # Global Decision
    max_p = max(powers)
    best_bs = (3 - np.argmax(powers)) if max_p > THRESHOLD else 0
    
    return jsonify({
        'results': results,
        'best_bs': int(best_bs),
        'snapped_x': float(user_coords[u_idx, 0]),
        'snapped_y': float(user_coords[u_idx, 1]),
        'bounds': {'min_x': float(min_x), 'max_x': float(max_x), 'min_y': float(min_y), 'max_y': float(max_y)}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)