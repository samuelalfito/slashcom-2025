import os
import shutil
import numpy as np
import librosa
import joblib
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# --- KONFIGURASI DAN INISIALISASI ---

# Mengabaikan peringatan
warnings.filterwarnings('ignore')

# Membuat aplikasi FastAPI
app = FastAPI(
    title="API Prediksi Emosi & Stres dari Suara",
    description="API ini menerima file audio (.wav) dan mengembalikan prediksi emosi serta tingkat stres.",
    version="1.0.0"
)

# --- MEMUAT MODEL DAN SCALER SAAT APLIKASI DIMULAI ---
# Model hanya dimuat sekali untuk efisiensi

BASE_DIR = os.path.dirname(__file__) 

try:
    print("--- Memuat Model dan Scaler ---")
    model_emotion = joblib.load(os.path.join(BASE_DIR, "model", "model_emotion.joblib"))
    model_stress = joblib.load(os.path.join(BASE_DIR, "model", "model_stress.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.joblib"))
    print("✅ Model dan scaler berhasil dimuat.")
except FileNotFoundError:
    print("❌ Error: File model atau scaler tidak ditemukan. Pastikan path sudah benar.")
    # Jika model tidak ada, aplikasi tidak bisa berjalan dengan baik.
    # Anda bisa memilih untuk menghentikan aplikasi di sini jika perlu.
    model_emotion = None
    model_stress = None
    scaler = None

emotion_labels_display = {0: 'Sedih', 1: 'Lelah', 2: 'Marah', 3: 'Cemas', 4: 'Netral'}

# --- FUNGSI EKSTRAKSI FITUR (SAMA SEPERTI SEBELUMNYA) ---
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
        return features
    except Exception as e:
        print(f"Error saat mengekstrak fitur: {e}")
        return None

# --- API ENDPOINT UNTUK PREDIKSI ---
# Ini adalah "pintu" dimana aplikasi lain bisa mengirim file

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Endpoint untuk prediksi emosi dan stres.
    - **Menerima**: file audio dalam format .wav
    - **Mengembalikan**: prediksi emosi dan tingkat stres dalam format JSON.
    """
    if not all([model_emotion, model_stress, scaler]):
        raise HTTPException(status_code=500, detail="Model tidak berhasil dimuat di server.")

    # Simpan file yang di-upload sementara agar bisa dibaca librosa
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Ekstrak fitur dari file audio yang disimpan sementara
        features = extract_features(file_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Tidak dapat memproses file audio.")

        # 2. Lakukan scaling pada fitur
        features_scaled = scaler.transform([features])

        # 3. Lakukan prediksi
        pred_emotion_idx = model_emotion.predict(features_scaled)[0]
        pred_emotion = emotion_labels_display.get(pred_emotion_idx, "Tidak Diketahui")
        
        pred_stress = model_stress.predict(features_scaled)[0]
        pred_stress_clamped = max(1.0, min(10.0, float(pred_stress)))

        # 4. Kembalikan hasil dalam format JSON
        return {
            "filename": file.filename,
            "predicted_emotion": pred_emotion,
            "predicted_stress_level": round(pred_stress_clamped, 1)
        }
    finally:
        # Selalu hapus file sementara setelah selesai
        if os.path.exists(file_path):
            os.remove(file_path)

# --- MENJALANKAN SERVER (UNTUK DEVELOPMENT) ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# local: uvicorn main_api:app --reload
# local-wifi: uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
# http://127.0.0.1:8000/predict/