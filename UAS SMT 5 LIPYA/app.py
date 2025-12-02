import streamlit as st
import joblib
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

# --- Load Model ---
try:
    # Coba load model, jika belum ada, tampilkan pesan error
    model = joblib.load('model_jantung.sav')
except FileNotFoundError:
    st.error("File 'model_jantung.sav' tidak ditemukan. Silakan jalankan 'train.py' terlebih dahulu.")
    st.stop()

# --- Judul Aplikasi ---
st.title("Sistem Prediksi Penyakit Jantung")
st.write("Silakan isi formulir di bawah ini dengan data klinis pasien.")

# --- Form Input (Layout Grid) ---
with st.form("prediction_form"):
    st.subheader("Data Pasien")
    
    # Baris 1
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur (Age)", min_value=20, max_value=100, value=50, step=1, help="Usia dalam tahun (20-100).")
    with col2:
        sex = st.selectbox("Jenis Kelamin (Sex)", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")

    # Baris 2
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Tekanan Darah Istirahat (Trestbps)", min_value=90, max_value=200, value=120, step=1, help="Dalam mm Hg (90-200).")
    with col4:
        chol = st.number_input("Kolesterol (Chol)", min_value=100, max_value=600, value=200, step=1, help="Dalam mg/dl (100-600).")

    # Baris 3
    col5, col6 = st.columns(2)
    with col5:
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (FBS)", options=[1, 0], format_func=lambda x: "Ya (True)" if x == 1 else "Tidak (False)")
    with col6:
        restecg = st.selectbox("Hasil EKG Istirahat (Restecg)", options=[0, 1, 2], 
                               format_func=lambda x: f"{x} - " + ("Normal" if x==0 else "Kelainan Gelombang ST-T" if x==1 else "Hipertrofi Ventrikel Kiri"),
                               help="0: Normal, 1: Kelainan ST-T, 2: Hipertrofi Ventrikel Kiri")
        
    # Baris 4
    col7, col8 = st.columns(2)
    with col7:
        thalach = st.number_input("Detak Jantung Maksimum (Thalach)", min_value=60, max_value=220, value=150, step=1, help="(60-220).")
    with col8:
        exang = st.selectbox("Nyeri Dada akibat Olahraga (Exang)", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    # Baris 5
    col9, col10 = st.columns(2)
    with col9:
        oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f", help="Nilai depresi ST (0.0 - 6.2).")
    with col10:
        slope = st.selectbox("Kemiringan ST (Slope)", options=[0, 1, 2], 
                             format_func=lambda x: f"{x} - " + ("Upsloping (Naik)" if x==0 else "Flat (Datar)" if x==1 else "Downsloping (Turun)"),
                             help="0: Naik, 1: Datar, 2: Turun")

    # Baris 6
    col11, col12 = st.columns(2)
    with col11:
        ca = st.selectbox("Jumlah Pembuluh Darah Utama (CA)", options=[0, 1, 2, 3, 4], help="Jumlah pembuluh darah yang diwarnai flourosopy (0-4).")
    with col12:
        thal = st.selectbox("Thalassemia (Thal)", options=[0, 1, 2, 3],
                            format_func=lambda x: f"{x} - " + ("Unknown" if x==0 else "Normal" if x==1 else "Fixed Defect" if x==2 else "Reversable Defect"),
                            help="1: Normal, 2: Cacat Tetap, 3: Cacat Bisa Dipulihkan")
        
    # Baris 7 (Tipe Nyeri Dada - ditaruh di akhir agar form terlihat rapi)
    cp = st.selectbox("Tipe Nyeri Dada (Chest Pain Type)", options=[0, 1, 2, 3], 
                      format_func=lambda x: f"Tipe {x} - " + ("Typical Angina" if x==0 else "Atypical Angina" if x==1 else "Non-anginal Pain" if x==2 else "Asymptomatic"))

    # Tombol Submit
    submitted = st.form_submit_button("Prediksi Hasil")

# --- Logika Prediksi ---
if submitted:
    # Membuat DataFrame dari input
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    # Melakukan prediksi
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Menampilkan Hasil
    st.divider()
    st.subheader("Hasil Prediksi")
    
    if prediction == 1:
        st.error(f"**TERINDIKASI Penyakit Jantung**")
        st.write(f"Probabilitas: {probability[1]*100:.1f}%")
        st.warning("Saran: Disarankan untuk segera berkonsultasi dengan dokter spesialis jantung untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"**SEHAT / Risiko Rendah**")
        st.write(f"Probabilitas Sehat: {probability[0]*100:.1f}%")
        st.info("Saran: Pertahankan gaya hidup sehat, pola makan seimbang, dan olahraga teratur.")