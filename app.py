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

# Load model
@st.cache_resource
def load_model():
    try:
        with open('model_klasifikasi_pengendara.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e1:
        st.warning(f"Method 1 failed: {e1}")
        try:
            with open('model_klasifikasi_pengendara.pkl', 'rb') as file:
                model = pickle.load(file, encoding='latin1')
            return model
        except Exception as e2:
            st.warning(f"Method 2 failed: {e2}")
            try:
                import joblib
                model = joblib.load('model_klasifikasi_pengendara.pkl')
                return model
            except Exception as e3:
                st.error(f"All loading methods failed: {e3}")
                return None

model = load_model()

if model is None:
    st.error("❌ Gagal memuat model! Silakan periksa file model.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# Sidebar untuk input data
st.sidebar.header("📊 Input Data Sensor")

st.sidebar.markdown("### 📈 Sensor Accelerometer")
acc_mean_x = st.sidebar.number_input("Acc Mean X", value=0.0, format="%.6f")
acc_mean_y = st.sidebar.number_input("Acc Mean Y", value=0.0, format="%.6f")
acc_mean_z = st.sidebar.number_input("Acc Mean Z", value=0.0, format="%.6f")
acc_cov_x = st.sidebar.number_input("Acc Cov X", value=0.0, format="%.6f")
acc_cov_y = st.sidebar.number_input("Acc Cov Y", value=0.0, format="%.6f")
acc_cov_z = st.sidebar.number_input("Acc Cov Z", value=0.0, format="%.6f")
acc_skew_x = st.sidebar.number_input("Acc Skew X", value=0.0, format="%.6f")
acc_skew_y = st.sidebar.number_input("Acc Skew Y", value=0.0, format="%.6f")
acc_skew_z = st.sidebar.number_input("Acc Skew Z", value=0.0, format="%.6f")
acc_kurt_x = st.sidebar.number_input("Acc Kurt X", value=0.0, format="%.6f")
acc_kurt_y = st.sidebar.number_input("Acc Kurt Y", value=0.0, format="%.6f")
acc_kurt_z = st.sidebar.number_input("Acc Kurt Z", value=0.0, format="%.6f")
acc_sum_x = st.sidebar.number_input("Acc Sum X", value=0.0, format="%.6f")
acc_sum_y = st.sidebar.number_input("Acc Sum Y", value=0.0, format="%.6f")
acc_sum_z = st.sidebar.number_input("Acc Sum Z", value=0.0, format="%.6f")
acc_min_x = st.sidebar.number_input("Acc Min X", value=0.0, format="%.6f")
acc_min_y = st.sidebar.number_input("Acc Min Y", value=0.0, format="%.6f")
acc_min_z = st.sidebar.number_input("Acc Min Z", value=0.0, format="%.6f")
acc_max_x = st.sidebar.number_input("Acc Max X", value=0.0, format="%.6f")
acc_max_y = st.sidebar.number_input("Acc Max Y", value=0.0, format="%.6f")
acc_max_z = st.sidebar.number_input("Acc Max Z", value=0.0, format="%.6f")
acc_var_x = st.sidebar.number_input("Acc Var X", value=0.0, format="%.6f")
acc_var_y = st.sidebar.number_input("Acc Var Y", value=0.0, format="%.6f")
acc_var_z = st.sidebar.number_input("Acc Var Z", value=0.0, format="%.6f")
acc_median_x = st.sidebar.number_input("Acc Median X", value=0.0, format="%.6f")
acc_median_y = st.sidebar.number_input("Acc Median Y", value=0.0, format="%.6f")
acc_median_z = st.sidebar.number_input("Acc Median Z", value=0.0, format="%.6f")
acc_std_x = st.sidebar.number_input("Acc Std X", value=0.0, format="%.6f")
acc_std_y = st.sidebar.number_input("Acc Std Y", value=0.0, format="%.6f")
acc_std_z = st.sidebar.number_input("Acc Std Z", value=0.0, format="%.6f")

st.sidebar.markdown("### 🌀 Sensor Gyroscope")
gyro_mean_x = st.sidebar.number_input("Gyro Mean X", value=0.0, format="%.6f")
gyro_mean_y = st.sidebar.number_input("Gyro Mean Y", value=0.0, format="%.6f")
gyro_mean_z = st.sidebar.number_input("Gyro Mean Z", value=0.0, format="%.6f")
gyro_cov_x = st.sidebar.number_input("Gyro Cov X", value=0.0, format="%.6f")
gyro_cov_y = st.sidebar.number_input("Gyro Cov Y", value=0.0, format="%.6f")
gyro_cov_z = st.sidebar.number_input("Gyro Cov Z", value=0.0, format="%.6f")
gyro_skew_x = st.sidebar.number_input("Gyro Skew X", value=0.0, format="%.6f")
gyro_skew_y = st.sidebar.number_input("Gyro Skew Y", value=0.0, format="%.6f")
gyro_skew_z = st.sidebar.number_input("Gyro Skew Z", value=0.0, format="%.6f")
gyro_sum_x = st.sidebar.number_input("Gyro Sum X", value=0.0, format="%.6f")
gyro_sum_y = st.sidebar.number_input("Gyro Sum Y", value=0.0, format="%.6f")
gyro_sum_z = st.sidebar.number_input("Gyro Sum Z", value=0.0, format="%.6f")
gyro_kurt_x = st.sidebar.number_input("Gyro Kurt X", value=0.0, format="%.6f")
gyro_kurt_y = st.sidebar.number_input("Gyro Kurt Y", value=0.0, format="%.6f")
gyro_kurt_z = st.sidebar.number_input("Gyro Kurt Z", value=0.0, format="%.6f")
gyro_min_x = st.sidebar.number_input("Gyro Min X", value=0.0, format="%.6f")
gyro_min_y = st.sidebar.number_input("Gyro Min Y", value=0.0, format="%.6f")
gyro_min_z = st.sidebar.number_input("Gyro Min Z", value=0.0, format="%.6f")
gyro_max_x = st.sidebar.number_input("Gyro Max X", value=0.0, format="%.6f")
gyro_max_y = st.sidebar.number_input("Gyro Max Y", value=0.0, format="%.6f")
gyro_max_z = st.sidebar.number_input("Gyro Max Z", value=0.0, format="%.6f")
gyro_var_x = st.sidebar.number_input("Gyro Var X", value=0.0, format="%.6f")
gyro_var_y = st.sidebar.number_input("Gyro Var Y", value=0.0, format="%.6f")
gyro_var_z = st.sidebar.number_input("Gyro Var Z", value=0.0, format="%.6f")
gyro_median_x = st.sidebar.number_input("Gyro Median X", value=0.0, format="%.6f")
gyro_median_y = st.sidebar.number_input("Gyro Median Y", value=0.0, format="%.6f")
gyro_median_z = st.sidebar.number_input("Gyro Median Z", value=0.0, format="%.6f")
gyro_std_x = st.sidebar.number_input("Gyro Std X", value=0.0, format="%.6f")
gyro_std_y = st.sidebar.number_input("Gyro Std Y", value=0.0, format="%.6f")
gyro_std_z = st.sidebar.number_input("Gyro Std Z", value=0.0, format="%.6f")

# Buat array fitur (60 fitur: 30 Acc + 30 Gyro)
features = [
    acc_mean_x, acc_mean_y, acc_mean_z,
    acc_cov_x, acc_cov_y, acc_cov_z,
    acc_skew_x, acc_skew_y, acc_skew_z,
    acc_kurt_x, acc_kurt_y, acc_kurt_z,
    acc_sum_x, acc_sum_y, acc_sum_z,
    acc_min_x, acc_min_y, acc_min_z,
    acc_max_x, acc_max_y, acc_max_z,
    acc_var_x, acc_var_y, acc_var_z,
    acc_median_x, acc_median_y, acc_median_z,
    acc_std_x, acc_std_y, acc_std_z,
    gyro_mean_x, gyro_mean_y, gyro_mean_z,
    gyro_cov_x, gyro_cov_y, gyro_cov_z,
    gyro_skew_x, gyro_skew_y, gyro_skew_z,
    gyro_sum_x, gyro_sum_y, gyro_sum_z,
    gyro_kurt_x, gyro_kurt_y, gyro_kurt_z,
    gyro_min_x, gyro_min_y, gyro_min_z,
    gyro_max_x, gyro_max_y, gyro_max_z,
    gyro_var_x, gyro_var_y, gyro_var_z,
    gyro_median_x, gyro_median_y, gyro_median_z,
    gyro_std_x, gyro_std_y, gyro_std_z
]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Ringkasan Input")
    
    feature_names = [
        "AccMeanX", "AccMeanY", "AccMeanZ", "AccCovX", "AccCovY", "AccCovZ",
        "AccSkewX", "AccSkewY", "AccSkewZ", "AccKurtX", "AccKurtY", "AccKurtZ",
        "AccSumX", "AccSumY", "AccSumZ", "AccMinX", "AccMinY", "AccMinZ",
        "AccMaxX", "AccMaxY", "AccMaxZ", "AccVarX", "AccVarY", "AccVarZ",
        "AccMedianX", "AccMedianY", "AccMedianZ", "AccStdX", "AccStdY", "AccStdZ",
        "GyroMeanX", "GyroMeanY", "GyroMeanZ", "GyroCovX", "GyroCovY", "GyroCovZ",
        "GyroSkewX", "GyroSkewY", "GyroSkewZ", "GyroSumX", "GyroSumY", "GyroSumZ",
        "GyroKurtX", "GyroKurtY", "GyroKurtZ", "GyroMinX", "GyroMinY", "GyroMinZ",
        "GyroMaxX", "GyroMaxY", "GyroMaxZ", "GyroVarX", "GyroVarY", "GyroVarZ",
        "GyroMedianX", "GyroMedianY", "GyroMedianZ", "GyroStdX", "GyroStdY", "GyroStdZ"
    ]
    
    df_input = pd.DataFrame({
        "Fitur": feature_names,
        "Nilai": features
    })
    st.dataframe(df_input, use_container_width=True, height=400)

with col2:
    st.subheader("🎯 Prediksi")
    
    if st.button("🔍 Klasifikasikan", type="primary", use_container_width=True):
        input_array = np.array(features).reshape(1, -1)
        
        try:
            prediction = model.predict(input_array)
            
            # Cek apakah model memiliki predict_proba
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_array)
            else:
                prediction_proba = None
            
            result_class = prediction[0]
            
            st.markdown("---")
            st.markdown("### Hasil Klasifikasi:")
            
            # Tentukan label kelas (sesuaikan dengan model Anda)
            class_labels = {
                0: "🚗 Pengendara Normal",
                1: "⚠️ Pengendara Agresif",
                2: "😴 Pengendara Mengantuk",
                3: "📱 Pengendara Terdistraksi"
            }
            
            colors = {
                0: "#2ecc71",  # Hijau
                1: "#e74c3c",  # Merah
                2: "#f39c12",  # Oranye
                3: "#9b59b6"   # Ungu
            }
            
            st.markdown(f"""
            <div style="background-color: {colors.get(result_class, '#3498db')}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="margin: 0; color: white;">{class_labels.get(result_class, f'Kelas {result_class}')}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if prediction_proba is not None:
                st.markdown("### 📊 Probabilitas:")
                proba_df = pd.DataFrame({
                    "Kelas": [class_labels.get(i, f"Kelas {i}") for i in range(len(prediction_proba[0]))],
                    "Probabilitas": prediction_proba[0]
                })
                st.dataframe(proba_df, use_container_width=True)
                
                st.markdown("### 📈 Visualisasi Probabilitas:")
                st.bar_chart(proba_df.set_index("Kelas"))
                
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {e}")

# Footer
st.markdown("---")
st.markdown("© 2024 - Aplikasi Klasifikasi Pengendara | Dibangun dengan Streamlit")
