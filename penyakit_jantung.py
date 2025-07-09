import numpy as np
import joblib
import streamlit as st
import pandas as pd

# Load the model (ensure this file exists and is trained correctly)
model = joblib.load(open('penyakit_jantung.pkl', 'rb'))

# ====== Styling with CSS ======
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #d6336c;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #d6336c;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        margin-top: 15px;
    }
    .result-box {
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #ced4da;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Title ======
st.markdown('<div class="title">🫀 Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)

# ====== Input Form ======
col1, col2 = st.columns(2)

with col1:
    male = st.number_input('Jenis Kelamin (1 = Laki-laki, 0 = Perempuan)', min_value=0, max_value=1, step=1)
with col2:
    age = st.number_input('Umur (dalam tahun)', min_value=1)

with col1:
    currentSmoker = st.number_input('Perokok Saat Ini? (1 = Ya, 0 = Tidak)', min_value=0, max_value=1, step=1)
with col2:
    prevalentHyp = st.number_input('Hipertensi? (1 = Ya, 0 = Tidak)', min_value=0, max_value=1, step=1)

with col1:
    diabetes = st.number_input('Diabetes? (1 = Ya, 0 = Tidak)', min_value=0, max_value=1, step=1)
with col2:
    totChol = st.number_input('Total Kolesterol', min_value=0.0)

with col1:
    heartRate = st.number_input('Denyut Jantung', min_value=0.0)
with col2:
    glucose = st.number_input('Kadar Glukosa Darah', min_value=0.0)

with col1:
    prevalentStroke = st.number_input('Riwayat Stroke? (1 = Ya, 0 = Tidak)', min_value=0, max_value=1, step=1)

# ====== Hasil Prediksi ======
heart_diagnosis = ''

if st.button('Hasil Prediksi Penyakit Jantung'):
    try:
        # Gather the inputs into a dictionary (with all 9 features)
        new_patient_data = {
            'male': [male],
            'age': [age],
            'currentSmoker': [currentSmoker],
            'prevalentHyp': [prevalentHyp],
            'diabetes': [diabetes],
            'totChol': [totChol],
            'heartRate': [heartRate],
            'glucose': [glucose],
            'prevalentStroke': [prevalentStroke]
        }

        # Convert the dictionary to a DataFrame
        new_patient_df = pd.DataFrame(new_patient_data)

        # Predict using the model
        heart_prediction = model.predict(new_patient_df)

        # Display result
        if heart_prediction[0] == 1:
            heart_diagnosis = '⚠️ Pasien Terkena Penyakit Jantung'
        else:
            heart_diagnosis = '✅ Pasien Tidak Terkena Penyakit Jantung'

        # Display the result in a styled box
        st.markdown(f'<div class="result-box">{heart_diagnosis}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Error dalam memproses prediksi: {e}')  # Error handling if the input format is incorrect
