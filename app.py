import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Klasifikasi Pengendara",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Aplikasi Klasifikasi Pengendara")
st.markdown("Aplikasi ini digunakan untuk mengklasifikasikan perilaku atau tipe pengendara berdasarkan data sensor (Akselerometer dan Gyroscope)")

# Load model dengan berbagai metode
@st.cache_resource
def load_model():
    try:
        # Method 1: Standard pickle load
        with open('model_klasifikasi_pengendara.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e1:
        st.warning(f"Method 1 failed: {e1}")
        try:
            # Method 2: With encoding parameter
            with open('model_klasifikasi_pengendara.pkl', 'rb') as file:
                model = pickle.load(file, encoding='latin1')
            return model
        except Exception as e2:
            st.warning(f"Method 2 failed: {e2}")
            try:
                # Method 3: Using joblib (better for scikit-learn models)
                import joblib
                model = joblib.load('model_klasifikasi_pengendara.pkl')
                return model
            except Exception as e3:
                st.error(f"All loading methods failed: {e3}")
                return None

model = load_model()

if model is None:
    st.error("""
    ❌ Gagal memuat model!
    
    **Kemungkinan penyebab:**
    1. Versi library tidak kompatibel
    2. File model corrupt
    3. Perbedaan arsitektur (Windows vs Linux)
    
    **Solusi:**
    - Pastikan file model_klasifikasi_pengendara.pkl ada di direktori yang sama
    - Coba train ulang model di environment dengan versi library yang konsisten
    - Gunakan Python 3.9 untuk deployment
    """)
    st.stop()

st.success("✅ Model berhasil dimuat!")

# [LANJUTKAN DENGAN KODE INPUT SEBELUMNYA...]
# (Sisanya sama seperti kode app.py sebelumnya)
