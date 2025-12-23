# -*- coding: utf-8 -*-
"""
Eksperimen_MSML (Healthcare)_Rahma_Putri.py
Otomasi Preprocessing Dataset Healthcare
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import sys

def automate_healthcare_preprocessing(input_path, output_path):
    print("🚀 Memulai proses preprocessing otomatis untuk Healthcare Dataset...\n")

    # --- DEBUGGING (Boleh dihapus nanti) ---
    print(f"📂 Lokasi Script berjalan: {os.getcwd()}")
    if os.path.exists(input_path):
        print(f"✅ File ditemukan di: {input_path}")
    else:
        print(f"❌ File TIDAK ditemukan di: {input_path}")
        print(f"📂 Cek isi folder root: {os.listdir('.')}")
        if os.path.exists('preprocessing'):
             print(f"📂 Cek isi folder preprocessing: {os.listdir('preprocessing')}")
    # ----------------------------------------

    # 1. Load dataset
    if not os.path.exists(input_path):
        print(f"❌ Error Fatal: File {input_path} tidak ditemukan! Menghentikan proses.")
        sys.exit(1) # Keluar supaya Action error

    df = pd.read_csv(input_path)
    print(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # 2. Menangani missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ Ditemukan {missing} nilai kosong, akan dihapus...")
        df.dropna(inplace=True)
    else:
        print("✅ Tidak ada missing values")

    # 3. Hapus duplikasi
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"⚠️ Ditemukan {dup} baris duplikat, akan dihapus...")
        df.drop_duplicates(inplace=True)
    else:
        print("✅ Tidak ada data duplikat")

    # 4. Encoding kolom kategorikal
    encoder = LabelEncoder()
    cat_cols = [
        'Gender', 'Blood Type', 'Medical Condition', 'Doctor', 
        'Hospital', 'Insurance Provider', 'Admission Type', 
        'Medication', 'Test Results'
    ]
    
    print("🔄 Melakukan encoding pada kolom kategorikal...")
    for col in cat_cols:
        if col in df.columns:
            df[f'{col}_Encoded'] = encoder.fit_transform(df[col])
            df.drop(columns=[col], inplace=True)
    
    cols_to_drop = ['Name', 'Date of Admission', 'Discharge Date']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # 5. Standarisasi kolom numerik
    scaler = StandardScaler()
    if 'Billing Amount' in df.columns:
        df['Billing_Amount_Scaled'] = scaler.fit_transform(df[['Billing Amount']])
        df.drop(columns=['Billing Amount'], inplace=True)
        print("✅ Kolom numerik 'Billing Amount' berhasil distandarisasi")

    # 6. Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 File hasil preprocessing disimpan di: {output_path}")
    print(f"📊 Ukuran akhir dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

if __name__ == "__main__":
    # PERUBAHAN ADA DISINI 👇
    # Karena file kamu ada di folder utama (root), kita hapus 'preprocessing/' dari path
    automate_healthcare_preprocessing(
        input_path='healthcare_dataset.csv', 
        output_path='preprocessed/healthcare_dataset_cleaned.csv' 
    )
