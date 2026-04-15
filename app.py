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
    except Exception:
        try:
            import joblib
            model = joblib.load('model_klasifikasi_pengendara.pkl')
            return model
        except Exception:
            return None

model = load_model()

if model is None:
    st.error("❌ Gagal memuat model! Silakan periksa file model.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# Cek library yang diperlukan untuk Excel
try:
    import openpyxl
    has_openpyxl = True
except ImportError:
    has_openpyxl = False
    st.warning("⚠️ Library 'openpyxl' tidak terinstall. Untuk membaca file Excel, silakan install dengan: pip install openpyxl")

# Sidebar untuk upload file
st.sidebar.header("📂 Upload Data Sensor")
st.sidebar.markdown("Upload file CSV atau Excel yang berisi data sensor dengan 60 fitur:")

# Daftar nama fitur yang diharapkan
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

uploaded_file = st.sidebar.file_uploader(
    "Pilih file (CSV atau Excel)", 
    type=['csv', 'xlsx', 'xls'],
    help="Upload file dengan 60 kolom fitur sesuai dengan yang diharapkan model"
)

# Informasi format file yang diharapkan
with st.sidebar.expander("📋 Format File yang Diharapkan"):
    st.markdown("""
    **Rekomendasi: Gunakan format CSV untuk menghindari error library**
    
    **Format 1 (CSV - Standar):**
    - Baris pertama: Nama kolom (60 fitur)
    - Baris berikutnya: Data per pengendara
    
    **Format 2 (Excel - Transpose seperti contoh):**
    - Kolom pertama: Nama fitur
    - Baris pertama: Header (bisa diabaikan)
    - Baris kedua: Nilai fitur
    
    **Cara konversi Excel ke CSV:**
    1. Buka file Excel
    2. File → Save As → Pilih CSV UTF-8
    3. Upload file CSV
    """)

# Fungsi untuk membaca file dengan berbagai format
def load_data(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Baca file CSV
            df = pd.read_csv(uploaded_file)
            return df
            
        elif file_extension in ['xlsx', 'xls']:
            # Cek apakah openpyxl tersedia
            if not has_openpyxl:
                st.error("❌ Library 'openpyxl' tidak terinstall. Silakan install dengan: pip install openpyxl")
                st.info("💡 Atau konversi file Excel ke CSV terlebih dahulu untuk menghindari error ini")
                return None
            
            # Coba baca Excel dengan berbagai metode
            try:
                # Metode 1: Baca tanpa header terlebih dahulu
                df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
                
                # Cek apakah file dalam format transpose (baris pertama berisi nama fitur)
                if df.shape[0] == 2 and df.shape[1] > 60:
                    # Format: baris pertama = nama fitur, baris kedua = nilai
                    st.info("📐 Mendeteksi format file transpose (2 baris x N kolom)")
                    feature_names_file = df.iloc[0].values
                    values = df.iloc[1].values
                    
                    # Buat dataframe dengan 1 baris
                    df_result = pd.DataFrame([values], columns=feature_names_file)
                    return df_result
                
                # Metode 2: Coba baca dengan header di baris pertama
                df = pd.read_excel(uploaded_file, header=0, engine='openpyxl')
                return df
                
            except Exception as e:
                st.error(f"Error membaca Excel: {e}")
                return None
        
        else:
            st.error(f"Format file {file_extension} tidak didukung")
            return None
            
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

# Fungsi untuk memeriksa dan mengkonversi tipe data
def convert_numeric(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    return df

# Fungsi untuk validasi fitur
def validate_features(df):
    # Cek apakah kolom yang ada cocok dengan feature_names
    df_columns = set(df.columns)
    required_columns = set(feature_names)
    
    missing_features = required_columns - df_columns
    if missing_features:
        st.error(f"❌ Kolom yang hilang: {list(missing_features)[:5]}... (total {len(missing_features)} kolom)")
        
        # Tampilkan kolom yang ada untuk debugging
        st.info("Kolom yang tersedia dalam file:")
        st.write(list(df.columns)[:10], "...")
        return False
    
    # Cek apakah ada kolom yang tidak diperlukan
    extra_features = df_columns - required_columns
    if extra_features:
        st.warning(f"⚠️ Kolom tambahan yang diabaikan: {list(extra_features)[:5]}... (total {len(extra_features)} kolom)")
    
    return True

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Data yang Diupload")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Konversi ke numeric
            df = convert_numeric(df)
            
            # Tampilkan informasi struktur file
            st.info(f"📊 Struktur file: {df.shape[0]} baris x {df.shape[1]} kolom")
            
            # Validasi fitur
            if validate_features(df):
                # Ambil hanya kolom yang diperlukan (urutkan sesuai feature_names)
                df_features = df[feature_names].copy()
                
                # Hapus baris yang memiliki nilai NaN
                before_drop = len(df_features)
                df_features = df_features.dropna()
                after_drop = len(df_features)
                
                if before_drop > after_drop:
                    st.warning(f"⚠️ Menghapus {before_drop - after_drop} baris yang memiliki nilai kosong (NaN)")
                
                st.success(f"✅ Data berhasil dimuat: {df_features.shape[0]} sampel, {df_features.shape[1]} fitur")
                
                # Tampilkan data
                st.dataframe(df_features.head(10), use_container_width=True)
                
                # Tampilkan statistik dasar
                with st.expander("📊 Statistik Deskriptif Data"):
                    st.dataframe(df_features.describe(), use_container_width=True)
                
                # Simpan ke session state untuk prediksi
                st.session_state['data'] = df_features
                st.session_state['original_data'] = df
            else:
                st.session_state['data'] = None
        else:
            st.session_state['data'] = None
    else:
        st.info("📂 Silakan upload file CSV atau Excel di sidebar untuk memulai klasifikasi")
        st.session_state['data'] = None

with col2:
    st.subheader("🎯 Prediksi")
    
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
    
    if st.button("🔍 Klasifikasikan Semua Data", type="primary", use_container_width=True, disabled=(uploaded_file is None)):
        if 'data' in st.session_state and st.session_state['data'] is not None:
            df_features = st.session_state['data']
            original_df = st.session_state.get('original_data', None)
            
            if len(df_features) == 0:
                st.error("❌ Tidak ada data yang valid untuk diklasifikasikan")
            else:
                try:
                    # Lakukan prediksi untuk semua data
                    with st.spinner('Sedang melakukan klasifikasi...'):
                        predictions = model.predict(df_features)
                    
                    # Cek apakah model memiliki predict_proba
                    if hasattr(model, 'predict_proba'):
                        predictions_proba = model.predict_proba(df_features)
                    else:
                        predictions_proba = None
                    
                    # Simpan hasil prediksi
                    result_df = df_features.copy()
                    result_df['Prediction'] = predictions
                    result_df['Class'] = [class_labels.get(p, f"Kelas {p}") for p in predictions]
                    
                    # Tampilkan hasil prediksi
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Klasifikasi:")
                    
                    # Ringkasan prediksi
                    pred_counts = pd.Series(predictions).value_counts()
                    st.markdown("#### Ringkasan Prediksi:")
                    
                    for class_id, count in pred_counts.items():
                        label = class_labels.get(class_id, f"Kelas {class_id}")
                        percentage = (count / len(predictions)) * 100
                        st.markdown(f"- {label}: **{count}** data ({percentage:.1f}%)")
                    
                    # Tampilkan tabel hasil
                    st.markdown("#### Detail Hasil Prediksi (10 data pertama):")
                    display_cols = ['Class'] + [col for col in df_features.columns[:3]]  # Tampilkan 3 fitur pertama
                    st.dataframe(result_df[display_cols].head(10), use_container_width=True)
                    
                    # Visualisasi distribusi prediksi
                    st.markdown("#### Distribusi Hasil Klasifikasi:")
                    chart_data = pd.DataFrame({
                        "Kelas": [class_labels.get(i, f"Kelas {i}") for i in pred_counts.index],
                        "Jumlah": pred_counts.values
                    })
                    st.bar_chart(chart_data.set_index("Kelas"))
                    
                    # Jika ada probabilitas, tampilkan untuk sampel
                    if predictions_proba is not None and st.checkbox("Tampilkan probabilitas untuk setiap data"):
                        st.markdown("#### Probabilitas per Data (5 data pertama):")
                        proba_cols = [class_labels.get(i, f"Kelas {i}") for i in range(predictions_proba.shape[1])]
                        proba_df = pd.DataFrame(predictions_proba[:5], columns=proba_cols)
                        proba_df.index = [f"Data {i+1}" for i in range(len(proba_df))]
                        st.dataframe(proba_df, use_container_width=True)
                    
                    # Tombol download hasil
                    output = result_df.copy()
                    if original_df is not None and len(original_df) == len(predictions):
                        # Gabungkan dengan data asli jika jumlah baris sama
                        output = original_df.copy()
                        output['Predicted_Class'] = predictions
                        output['Predicted_Label'] = [class_labels.get(p, f"Kelas {p}") for p in predictions]
                    
                    csv_output = output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv_output,
                        file_name="hasil_klasifikasi_pengendara.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Tampilkan pesan sukses
                    st.balloons()
                    st.success(f"✅ Berhasil mengklasifikasikan {len(predictions)} data!")
                    
                except Exception as e:
                    st.error(f"Error saat melakukan prediksi: {e}")
        else:
            st.warning("⚠️ Silakan upload file terlebih dahulu")
    
    # Prediksi untuk satu baris (opsional)
    if uploaded_file is not None and 'data' in st.session_state and st.session_state['data'] is not None:
        st.markdown("---")
        st.markdown("### 🔍 Prediksi Data Tertentu")
        
        df_features = st.session_state['data']
        if len(df_features) > 0:
            row_index = st.number_input("Pilih nomor baris data", min_value=0, max_value=len(df_features)-1, value=0, step=1)
            
            if st.button("Klasifikasikan Data Ini", use_container_width=True):
                try:
                    single_data = df_features.iloc[[row_index]]
                    prediction = model.predict(single_data)[0]
                    
                    st.markdown(f"""
                    <div style="background-color: {colors.get(prediction, '#3498db')}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 10px;">
                        <h3 style="margin: 0; color: white;">Data ke-{row_index + 1}</h3>
                        <h2 style="margin: 10px 0 0 0; color: white;">{class_labels.get(prediction, f'Kelas {prediction}')}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(single_data)[0]
                        st.markdown("**Probabilitas:**")
                        for i, p in enumerate(proba):
                            st.progress(float(p), text=f"{class_labels.get(i, f'Kelas {i}')}: {p:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("© 2024 - Aplikasi Klasifikasi Pengendara | Dibangun dengan Streamlit")
