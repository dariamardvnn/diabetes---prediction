
import streamlit as st
import joblib

st.title("Prediksi Kesehatan")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.write("Aplikasi berhasil berjalan!")
