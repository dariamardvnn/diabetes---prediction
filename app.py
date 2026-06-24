import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Aplikasi Prediksi Diabetes")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan diabetes menggunakan Machine Learning.")

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")          # ganti sesuai nama file kamu
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ==================== INPUT FORM ====================
st.subheader("📋 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Usia (Age)", min_value=10, max_value=100, value=30, step=1)

# ==================== PREDIKSI ====================
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    
    # Buat DataFrame dengan urutan fitur yang SAMA persis seperti training
    input_data = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])
    
    # Scaling
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    
    # Probabilitas (jika model mendukung)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]
    else:
        prob = None
    
    # ==================== HASIL ====================
    st.subheader("📊 Hasil Prediksi")
    
    if prediction == 1:
        st.error("⚠️ **Hasil: Kemungkinan Diabetes Positif**")
    else:
        st.success("✅ **Hasil: Kemungkinan Diabetes Negatif**")
    
    if prob is not None:
        st.metric(label="Probabilitas Diabetes", value=f"{prob*100:.2f}%")
    
    st.caption("Catatan: Hasil ini hanya prediksi berbasis machine learning. Silakan konsultasikan dengan dokter untuk diagnosis yang akurat.")

# ==================== INFO TAMBAHAN ====================
with st.expander("ℹ️ Informasi Model"):
    st.write("""
    - **Model**: Logistic Regression (atau model terbaik yang kamu simpan)
    - **Jumlah Fitur**: 8 fitur
    - **Scaler**: StandardScaler
    - **Dataset Training**: Pima Indians Diabetes
    """)