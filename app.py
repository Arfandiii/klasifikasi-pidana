"""
SISTEM KLASIFIKASI KASUS KRIMINAL DENGAN PREPROCESSING SASTRAWI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import re
import warnings
import subprocess
import sys
warnings.filterwarnings('ignore')

# ==================== CEK DAN INSTAL DEPENDENCIES ====================
def install_packages():
    """Install required packages"""
    required_packages = [
        'openpyxl',  # Untuk membaca file Excel
        'Sastrawi',   # Untuk preprocessing Bahasa Indonesia
    ]
    
    installed_packages = []
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            installed_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    return installed_packages, missing_packages

# Tampilkan status packages di sidebar
st.sidebar.title("📦 Status Dependencies")

installed, missing = install_packages()

if installed:
    st.sidebar.success("✅ Packages terinstal:")
    for pkg in installed:
        st.sidebar.text(f"  - {pkg}")

if missing:
    st.sidebar.warning("⚠️ Packages yang diperlukan:")
    for pkg in missing:
        st.sidebar.text(f"  - {pkg}")
    
    if st.sidebar.button("🔧 Install Packages"):
        with st.sidebar:
            with st.spinner(f"Menginstal {len(missing)} packages..."):
                for package in missing:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                        st.success(f"✅ {package} berhasil diinstal")
                    except:
                        st.error(f"❌ Gagal menginstal {package}")
                st.rerun()

# ==================== IMPORT SASTRAWI SETELAH INSTALASI ====================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    
    # Inisialisasi Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    if 'Sastrawi' not in missing:
        st.sidebar.warning("Sastrawi tidak tersedia. Menggunakan preprocessing dasar.")

# ==================== KONFIGURASI APLIKASI ====================
st.set_page_config(
    page_title="Sistem Klasifikasi Kasus Kriminal",
    page_icon="🔍",
    layout="wide"
)

# ==================== FUNGSI PREPROCESSING DENGAN SASTRAWI ====================
def preprocess_text_indonesia(text, use_sastrawi=True):
    """
    Fungsi preprocessing teks Bahasa Indonesia dengan Sastrawi
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 1. Case folding (ubah ke huruf kecil)
    text = text.lower()
    
    # 2. Hapus karakter khusus dan angka
    text = re.sub(r'[^\w\s]', ' ', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)       # Hapus angka
    text = re.sub(r'\s+', ' ', text)      # Hapus spasi berlebih
    text = text.strip()
    
    if use_sastrawi and SASTRAWI_AVAILABLE:
        try:
            # 3. Hapus stopwords Bahasa Indonesia
            text = stopword_remover.remove(text)
            
            # 4. Stemming (mengubah kata ke bentuk dasar)
            text = stemmer.stem(text)
        except Exception as e:
            st.warning(f"Error dalam preprocessing Sastrawi: {e}")
    
    return text

# ==================== FUNGSI UTAMA ====================
def main():
    # Judul aplikasi
    st.title("🔍 Sistem Klasifikasi Kasus Kriminal")
    st.markdown("""
    Sistem ini menggunakan algoritma **Naive Bayes** dengan pembobotan **TF-IDF** 
    dan **preprocessing Sastrawi** untuk Bahasa Indonesia.
    """)
    
    # Sidebar untuk navigasi dan konfigurasi
    st.sidebar.title("⚙️ Konfigurasi")
    
    # Opsi preprocessing
    use_sastrawi = st.sidebar.checkbox("Gunakan Sastrawi", value=True, 
                                       disabled=not SASTRAWI_AVAILABLE,
                                       help="Gunakan stemming dan stopword removal Bahasa Indonesia")
    
    # Opsi upload file
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload file Excel dengan data kasus", 
        type=['xlsx', 'xls'],
        help="File harus memiliki kolom 'URAIAN SINGKAT (MO)' dan 'PERKARA'"
    )
    
    st.sidebar.markdown("---")
    menu = st.sidebar.selectbox(
        "📋 Menu Navigasi",
        ["📊 Dashboard", "🔮 Prediksi Kasus", "📚 Data & Preprocessing", "📈 Analisis Model", "📘 Panduan"]
    )
    
    # Load data
    if uploaded_file is not None:
        df = load_data_from_upload(uploaded_file)
    else:
        df = load_sample_data()
    
    if df is None:
        st.error("Tidak dapat memuat data.")
        return
    
    # Training model (cached dengan parameter preprocessing)
    model_data = train_model(df, use_sastrawi)
    
    if menu == "📊 Dashboard":
        show_dashboard(df)
    elif menu == "🔮 Prediksi Kasus":
        show_prediction(df, model_data, use_sastrawi)
    elif menu == "📚 Data & Preprocessing":
        show_preprocessing(df, use_sastrawi)
    elif menu == "📈 Analisis Model":
        show_model_analysis(df, model_data)
    elif menu == "📘 Panduan":
        show_guide()

# ==================== FUNGSI BANTU ====================
def load_data_from_upload(uploaded_file):
    """Memuat data dari file yang diupload"""
    try:
        df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"✅ Data berhasil dimuat: {len(df)} baris")
        
        # Cek kolom yang diperlukan
        required_columns = ['URAIAN SINGKAT (MO)', 'PERKARA']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
            st.info("Pastikan file memiliki kolom: 'URAIAN SINGKAT (MO)' dan 'PERKARA'")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_sample_data():
    """Membuat dataframe contoh"""
    st.info("Menggunakan data contoh. Upload file Excel Anda sendiri untuk menggunakan data aktual.")
    
    data = [
        [1, "LP/B/01/I/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 03-Jan-2024", "Penganiayaan", "Sungai Pinyuh", "Tya Amelliani", "Heriyadi", "Pelaku melakukan Penganiayaan dengan memukul dan menendang pelapor", "–", "Sidik", "P.21B-631/O.1.5/Eoh.1/02/2024,tgl.28-02-2024. Tahap II B/41/II/Res.1.6/2024,Tgl. 29-02-2024."],
        [2, "LP/B/02/I/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 14-Jan-2024", "Pencurian", "Sungai Pinyuh", "Ayu Mufidatun Hasanah", "M. Taufik, dkk", "Pelaku masuk kedalam rumah dengan mencongkel pintu masuk mengambil laptop", "1(satu) buah laptop merk ASUS", "Sidik", "P.21B-594/O.1.5/Eoh.1/02/2024,tgl.22-02-2024. Tahap II B/51/III/Res.1.8/2024,Tgl. 14-03-2024."],
        [3, "LP/B/03/II/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 02-Feb-2024", "Pencurian", "Sungai Pinyuh", "Ainul Yaqin", "Jumri Gunawan", "Pelaku masuk kedalam rumah melalui jendela mengambil 1 (satu) HP", "1(satu) buah Kotak HP, 1(satu) buah HP merk Vivo S1 Pro 1920 warna crystal blue", "Sidik", "P.21B-872/O.1.5/Eoh.1/03/2024,tgl.27-03-2024. Tahap II B/63/III/Res.1.8/2024,Tgl. 27-03-2024."],
        [4, "LP/B/04/III/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 11-Maret-2024", "Pencurian", "Desa Galang, Kec. Sungai Pinyuh", "Misnadi", "Liau Siun Phin", "Pelaku melakukan pencurian mengambil HP yang diletakan diatas meja saat pelapor bekerja menebang pohon", "1(satu) buah Kotak HP, 1(satu) buah HP merk Redme C51 warna hitam", "Sidik", "P.21B-1138/O.1.5/Eoh.1/05/2024,tgl.03-05-2024. Tahap II B/85/V/Res.1.8/2024,Tgl. 07-05-2024."],
        [5, "LP/B/05/IV/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 29-April-2024", "Pencurian", "Sungai Pinyuh", "Rachmawati", "Alfian", "Pelaku melakukan pencurian dengan membukan kunci pintu dan mengambil 2 (dua) buah HP", "1(satu) buah Kotak HP, 1(satu) buah HP merk Iphone 7 plus warna black matta, 1(satu) buah HP merk VIVO Y91 warna biru", "Sidik", "P.21B-1620/O.1.5/Eoh.1/06/2024,tgl.26-06-2024. Tahap II B/103/VI/Res.1.8/2024,Tgl. 27-06-2024."],
        [6, "LP/B/06/VI/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 02-Juni-2024", "Pencurian", "Sungai Pinyuh", "Abdul Aziz", "Perdiansyah", "Pelaku bersama-sama melakukan pencurian 2 (dua) buah aki kering yang terpasang dikendaraan dump truk yang diparkir digarasi", "1 (satu) buah Aki kering merk XTRA AMF 55D 26 R 12V 60AH warna hitam, 1 (satu) buah Aki kering merk TAG 48D 26R 12V 50AH warna hitam", "Sidik", "P.21B-1836/O.1.5/Eoh.1/07/2024,tgl.17-07-2024. Tahap II B/126/VII/Res.1.8/2024,Tgl. 30-07-2024."],
        [7, "LP/B/07/VI/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 02-Juni-2024", "Pencurian", "Sungai Pinyuh", "Rudi Sugianto", "Achmad Albie Rossi Ramadana dkk", "Pelaku bersama-sama melakukan pencurian dengan masuk kedalam gudang mencongkel dinding papan dan mengambil 1 (satu) buah timbangan gantung", "1 (satu) buah flash disk berisi rekaman CCTV, 1 (satu) buah timbangan gantung 110 Kg", "Sidik", "P.21B-2042/O.1.5/Eoh.1/03/2024,tgl.13-08-2024. Tahap II B/148/IX/Res.1.8/2024,Tgl. 03-09-2024."],
        [8, "LP/B/08/VI/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 12-Juni-2024", "Pencurian", "Desa Sungai Purun Kecil, Kec. Sungai Pinyuh", "Nurul Utami", "Teguh Angrian dkk", "pelaku bersama-sama melakukan pencurian sepeda motor yang diparkir ditempat pencucian sepeda motor", "1 (satu) lbr STNK sepeda motor Honda Scoopy KB 6270 BN, 1 (satu) buah card reader berisi rekaman CCTV, 1 (satu) unit sepeda motor Honda Scoopy warna hitam", "Sidik", "P.21B-1785/O.1.5/Eoh.1/07/2024,tgl.15-07-2024. Tahap II B/133/VIII/Res.1.8/2024,Tgl. 01-08-2024."],
        [9, "LP/B/09/VII/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 30-Juli-20284", "Pencurian", "Desa Galang, Kec. Sungai Pinyuh", "Kau Kui", "Rusdy Als Aliung", "Pelaku mengambil terpal, mesin pompa air, timbangan digital dan kopling selang pemadam dalam gudang yang tidak terkunci", "1 (satu) lbr terpal merk ORCHID warna hijau, 1 (satu) buah timbangan digital merk MATRIX uk 30 Kg", "Sidik", "P.21B-2547/O.1.5/Eoh.1/10/2024,tgl.23-09-2024. Tahap II B/157/IX/Res.1.8/2024,Tgl. 26-09-2024."],
        [10, "LP/B/10/IX/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 08-Sep-2024", "Pencurian", "Desa Peniraman, Kec. Sungai Pinyuh", "H. Usman Ghazali Als H. Margi", "Danil Als Kecap", "Pelaku melakukan pencurian 2 (dua) buah aki kering yang terpasang dikendaraan dump truk yang diparkir dihalaman rumah", "1 (satu) buah Aki kering merk YUASA warna hitam, 1 (satu) buah Aki kering merk GS ASTRA warna hitam, 1 (satu) buah kunci inggris, 1 (satu) buah obeng, 1 lbr karung plastik warna putih, 1 (satu) unit sepeda motor Suzuki Shogun KB 2693 BP warna hitam", "Sidik", "P.21B-2810/O.1.5/Eoh.1/10/2024,tgl.15-07-2024. Tahap II B/172/XI/Res.1.8/2024,Tgl. 5-11-2024."],
        [11, "LP/B/11/XI/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 14-Nov-2024", "Pencurian dengan kekerasan", "Jl. Raya Sui Bakau Besar Laut", "Syahroni", "Heirul Beri Als Hairul", "Pelaku datang menyemprotkan cairan cabai dan memukul korban kemudian merampas tas berisi uang", "1 (satu) helai celana pendek warna hitam, 1 (satu) unit sepeda motor Yamaha Jupiter MX tanpa plat nomor warna hitam lis biru, 2 (dua) buah plat nomor KB 2656 SZ, 1 (satu) buah Helm GM warna hitam, 1 (satu) buah botol semprot plastik berisikan cairan cabai, Uang tunai sebesar Rp.5.850.000, (lima juta delapan ratus lima puluh ribu rupiah)", "Sidik", "P.21B-97/O.1.5/Eoh.1/01/2025,tgl.08-01-2025. Tahap II B/05/I/Res.1.8/2025,Tgl. 09-01-2025."],
        [12, "LP/B/12/XII/2024/Kalbar/Res Mpw/Sek Sui Pinyuh tgl. 04-Des-2024", "Penganiayaan ringan", "Peniraman", "Hj. Maliyeh binti Marsai Slimin", "Muryuki Als H. Marzuki bin Marsai Slimin", "Pelaku datang mendorong korban kemudian meremas wajah dan leher korban", "-", "-", "Tipiring sidang tgl. 11-12-2024"]
    ]
    
    columns = ["NO", "NO & TGL LAPORAN", "PERKARA", "TKP", "PELAPOR", "TERLAPOR", "URAIAN SINGKAT (MO)", "BARANG BUKTI", "PROSES", "Keterangan"]
    return pd.DataFrame(data, columns=columns)

@st.cache_resource
def train_model(df, use_sastrawi=True):
    """Melatih model klasifikasi dengan preprocessing"""
    # Preprocessing dengan Sastrawi
    df_text = df[['URAIAN SINGKAT (MO)', 'PERKARA']].copy()
    
    # Proses teks dengan preprocessing
    if use_sastrawi and SASTRAWI_AVAILABLE:
        with st.spinner("Melakukan preprocessing teks dengan Sastrawi..."):
            df_text['cleaned_text'] = df_text['URAIAN SINGKAT (MO)'].apply(
                lambda x: preprocess_text_indonesia(x, use_sastrawi=True)
            )
    else:
        df_text['cleaned_text'] = df_text['URAIAN SINGKAT (MO)'].apply(
            lambda x: preprocess_text_indonesia(x, use_sastrawi=False)
        )
    
    # Label encoding
    label_mapping = {label: i for i, label in enumerate(df_text['PERKARA'].unique())}
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    df_text['label'] = df_text['PERKARA'].map(label_mapping)
    
    # TF-IDF dengan parameter untuk Bahasa Indonesia
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words=None  # Kita sudah menghapus stopwords di preprocessing
    )
    
    X = vectorizer.fit_transform(df_text['cleaned_text'])
    y = df_text['label']
    
    # Model Naive Bayes
    model = MultinomialNB(alpha=1.0)
    model.fit(X, y)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'label_mapping': label_mapping,
        'reverse_mapping': reverse_mapping,
        'X': X,
        'y': y,
        'df_text': df_text
    }

# ==================== HALAMAN DASHBOARD ====================
def show_dashboard(df):
    """Menampilkan dashboard utama"""
    st.header("📊 Dashboard Analisis Kasus Kriminal")
    
    # Statistik
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Kasus", len(df))
    
    with col2:
        st.metric("Jenis Perkara", len(df['PERKARA'].unique()))
    
    with col3:
        pencurian_count = len(df[df['PERKARA'].str.contains('Pencurian')])
        st.metric("Kasus Pencurian", pencurian_count)
    
    with col4:
        tkp_mode = df['TKP'].mode()[0] if len(df['TKP'].mode()) > 0 else "-"
        st.metric("TKP Paling Banyak", tkp_mode)
    
    # Visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Jenis Perkara")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Hitung distribusi
        crime_dist = df['PERKARA'].value_counts()
        
        # Buat bar plot
        bars = ax.bar(range(len(crime_dist)), crime_dist.values, color='skyblue')
        ax.set_xticks(range(len(crime_dist)))
        ax.set_xticklabels(crime_dist.index, rotation=45, ha='right')
        ax.set_xlabel('Jenis Perkara')
        ax.set_ylabel('Jumlah Kasus')
        ax.grid(axis='y', alpha=0.3)
        
        # Tambah label nilai
        for bar, count in zip(bars, crime_dist.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Lokasi Kejadian (Top 5)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Hitung top 5 TKP
        tkp_counts = df['TKP'].value_counts().head(5)
        
        # Buat pie chart
        ax.pie(tkp_counts.values, labels=tkp_counts.index, autopct='%1.1f%%',
              startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(tkp_counts))))
        ax.set_title('Top 5 Tempat Kejadian Perkara')
        
        st.pyplot(fig)
    
    # Tabel data terbaru
    st.subheader("Data Kasus Kriminal Terbaru")
    st.dataframe(
        df[['NO', 'PERKARA', 'TKP', 'PELAPOR', 'TERLAPOR', 'URAIAN SINGKAT (MO)']].head(10),
        use_container_width=True
    )

# ==================== HALAMAN PREDIKSI ====================
def show_prediction(df, model_data, use_sastrawi):
    """Menampilkan halaman prediksi"""
    st.header("🔮 Prediksi Jenis Perkara Baru")
    
    # Informasi model
    with st.expander("ℹ️ Informasi Model"):
        st.info(f"""
        **Konfigurasi Model:**
        - Algoritma: Naive Bayes Multinomial
        - Feature Extraction: TF-IDF
        - Preprocessing: {'Dengan Sastrawi' if use_sastrawi else 'Tanpa Sastrawi'}
        - Jumlah Data Training: {len(df)} kasus
        - Jumlah Kategori: {len(model_data['label_mapping'])}
        """)
    
    # Input pengguna
    st.subheader("📝 Masukkan Deskripsi Kasus")
    
    # Contoh cepat
    example_options = {
        "Pencurian HP": "Pelaku masuk ke dalam rumah dan mencuri handphone",
        "Penganiayaan": "Pelaku memukul korban dengan tangan kosong",
        "Pencurian dengan Kekerasan": "Pelaku menyemprotkan cairan cabai dan merampas tas korban",
        "Penganiayaan Ringan": "Pelaku mendorong dan meremas wajah korban",
        "Pencurian Kendaraan": "Pelaku mencuri sepeda motor yang diparkir di garasi",
        "Pencurian Aki": "Pelaku mencuri aki kering dari kendaraan yang diparkir"
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_example = st.selectbox(
            "Pilih contoh kasus atau tulis sendiri:",
            ["-- Tulis deskripsi sendiri --"] + list(example_options.keys())
        )
    
    with col2:
        if st.button("🔄 Reset"):
            st.rerun()
    
    if selected_example in example_options:
        input_text = st.text_area(
            "Deskripsi kasus:",
            value=example_options[selected_example],
            height=150,
            help="Deskripsikan kejadian dengan jelas dalam Bahasa Indonesia"
        )
    else:
        input_text = st.text_area(
            "Deskripsi kasus:",
            placeholder="Contoh: pelaku memasuki rumah dengan membongkar jendela dan mencuri handphone...",
            height=150
        )
    
    # Tombol prediksi
    if st.button("🔍 Analisis Kasus", type="primary", use_container_width=True) and input_text.strip():
        with st.spinner("Menganalisis kasus..."):
            # Preprocessing input
            cleaned_text = preprocess_text_indonesia(input_text, use_sastrawi)
            
            # Tampilkan teks setelah preprocessing
            with st.expander("📝 Lihat hasil preprocessing"):
                st.write("**Teks asli:**")
                st.write(input_text)
                st.write("**Setelah preprocessing:**")
                st.write(cleaned_text)
            
            # Transformasi TF-IDF
            input_vector = model_data['vectorizer'].transform([cleaned_text])
            
            # Prediksi
            model = model_data['model']
            prediction_idx = model.predict(input_vector)[0]
            probabilities = model.predict_proba(input_vector)[0]
            confidence = max(probabilities)
            
            # Hasil prediksi
            st.subheader("📋 Hasil Analisis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Jenis Perkara",
                    model_data['reverse_mapping'][prediction_idx],
                    delta=f"{confidence:.2%} keyakinan"
                )
            
            with col2:
                status = "🎯 Tinggi" if confidence > 0.7 else "⚠️ Sedang" if confidence > 0.5 else "📉 Rendah"
                st.metric("Status Prediksi", status)
            
            with col3:
                # Hitung peringkat probabilitas
                sorted_probs = np.argsort(probabilities)[::-1]
                top3 = [model_data['reverse_mapping'][i] for i in sorted_probs[:3]]
                st.metric("Top 3 Kategori", ", ".join(top3[:2]))
            
            # Distribusi probabilitas
            st.subheader("📊 Distribusi Probabilitas")
            
            # Buat dataframe probabilitas
            prob_data = []
            for i, prob in enumerate(probabilities):
                if prob > 0.001:  # Hanya tampilkan probabilitas signifikan
                    prob_data.append({
                        'Jenis Perkara': model_data['reverse_mapping'][i],
                        'Probabilitas': prob,
                        'Peringkat': np.where(sorted_probs == i)[0][0] + 1
                    })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('Probabilitas', ascending=False)
            
            # Tampilkan dalam dua kolom
            col1, col2 = st.columns(2)
            
            with col1:
                # Tampilkan chart
                fig, ax = plt.subplots(figsize=(10, 4))
                
                colors = []
                for i in range(len(prob_df)):
                    if prob_df.iloc[i]['Peringkat'] == 1:
                        colors.append('lightgreen')
                    elif prob_df.iloc[i]['Peringkat'] <= 3:
                        colors.append('lightblue')
                    else:
                        colors.append('lightgray')
                
                bars = ax.barh(prob_df['Jenis Perkara'], prob_df['Probabilitas'], color=colors)
                ax.set_xlabel('Probabilitas')
                ax.set_title('Distribusi Probabilitas per Kategori')
                ax.set_xlim(0, 1)
                
                # Tambah nilai pada bar
                for bar, prob in zip(bars, prob_df['Probabilitas']):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
            
            with col2:
                # Tampilkan tabel probabilitas
                st.dataframe(
                    prob_df[['Peringkat', 'Jenis Perkara', 'Probabilitas']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Fitur penting yang mempengaruhi prediksi
            st.subheader("🔑 Kata Kunci yang Mempengaruhi Prediksi")
            
            # Dapatkan fitur penting untuk kelas yang diprediksi
            feature_names = model_data['vectorizer'].get_feature_names_out()
            feature_log_probs = model.feature_log_prob_[prediction_idx]
            
            # Ambil top 10 fitur positif dan negatif
            top_n = 10
            top_indices = np.argsort(feature_log_probs)[-top_n:][::-1]
            bottom_indices = np.argsort(feature_log_probs)[:top_n]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Kata kunci pendukung:**")
                features_data_pos = []
                for idx in top_indices:
                    if feature_log_probs[idx] > -10:
                        features_data_pos.append({
                            'Kata Kunci': feature_names[idx],
                            'Bobot': feature_log_probs[idx]
                        })
                
                if features_data_pos:
                    features_df_pos = pd.DataFrame(features_data_pos)
                    st.dataframe(features_df_pos, use_container_width=True)
                else:
                    st.write("Tidak ada kata kunci pendukung yang signifikan")
            
            with col2:
                st.write("**Kata kunci yang mengurangi:**")
                features_data_neg = []
                for idx in bottom_indices:
                    if feature_log_probs[idx] < 0:
                        features_data_neg.append({
                            'Kata Kunci': feature_names[idx],
                            'Bobot': feature_log_probs[idx]
                        })
                
                if features_data_neg:
                    features_df_neg = pd.DataFrame(features_data_neg)
                    st.dataframe(features_df_neg, use_container_width=True)
                else:
                    st.write("Tidak ada kata kunci yang mengurangi")
            
            # Simpan riwayat
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.insert(0, {
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Deskripsi': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                'Prediksi': model_data['reverse_mapping'][prediction_idx],
                'Keyakinan': confidence
            })
    
    # Riwayat prediksi
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.subheader("📜 Riwayat Prediksi")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hapus Riwayat"):
                st.session_state.prediction_history = []
                st.rerun()
        with col2:
            if st.button("Ekspor ke CSV"):
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="riwayat_prediksi.csv",
                    mime="text/csv"
                )

# ==================== HALAMAN PREPROCESSING ====================
def show_preprocessing(df, use_sastrawi):
    """Menampilkan halaman preprocessing data"""
    st.header("📚 Data & Preprocessing")
    
    # Pilih contoh teks untuk preprocessing demo
    st.subheader("🎯 Demo Preprocessing Teks")
    
    sample_texts = df['URAIAN SINGKAT (MO)'].head(5).tolist()
    
    selected_idx = st.selectbox(
        "Pilih contoh teks untuk melihat proses preprocessing:",
        range(len(sample_texts)),
        format_func=lambda x: f"Contoh {x+1}: {sample_texts[x][:80]}..."
    )
    
    sample_text = sample_texts[selected_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Teks Asli:**")
        st.info(sample_text)
    
    with col2:
        st.write("**Setelah Preprocessing:**")
        processed_text = preprocess_text_indonesia(sample_text, use_sastrawi)
        st.success(processed_text)
    
    # Detail preprocessing
    with st.expander("🔍 Detail Proses Preprocessing"):
        st.write("**Langkah-langkah preprocessing:**")
        steps = [
            "1. **Case Folding**: Mengubah semua huruf menjadi huruf kecil",
            "2. **Cleaning**: Menghapus tanda baca, angka, dan karakter khusus",
            "3. **Normalisasi Spasi**: Menghapus spasi berlebih",
            "4. **Stopword Removal** (Sastrawi): Menghapus kata penghubung yang umum",
            "5. **Stemming** (Sastrawi): Mengubah kata ke bentuk dasarnya"
        ]
        
        for step in steps:
            st.write(step)
    
    # Tabel data dengan kolom preprocessing
    st.subheader("📊 Data dengan Hasil Preprocessing")
    
    # Buat dataframe dengan teks yang sudah dipreprocess
    df_preprocessed = df.copy()
    df_preprocessed['Teks Setelah Preprocessing'] = df_preprocessed['URAIAN SINGKAT (MO)'].apply(
        lambda x: preprocess_text_indonesia(x, use_sastrawi)
    )
    
    # Pilih kolom untuk ditampilkan
    columns_to_show = st.multiselect(
        "Pilih kolom untuk ditampilkan:",
        df_preprocessed.columns.tolist(),
        default=['NO', 'PERKARA', 'TKP', 'URAIAN SINGKAT (MO)', 'Teks Setelah Preprocessing']
    )
    
    if columns_to_show:
        st.dataframe(
            df_preprocessed[columns_to_show],
            use_container_width=True,
            height=400
        )
    
    # Statistik teks
    st.subheader("📈 Statistik Teks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Hitung jumlah kata sebelum preprocessing
        word_counts_before = df['URAIAN SINGKAT (MO)'].apply(
            lambda x: len(str(x).split())
        )
        avg_words_before = word_counts_before.mean()
        st.metric("Rata-rata kata (sebelum)", f"{avg_words_before:.1f}")
    
    with col2:
        # Hitung jumlah kata setelah preprocessing
        word_counts_after = df_preprocessed['Teks Setelah Preprocessing'].apply(
            lambda x: len(str(x).split())
        )
        avg_words_after = word_counts_after.mean()
        st.metric("Rata-rata kata (sesudah)", f"{avg_words_after:.1f}")
    
    with col3:
        # Hitung reduksi kata
        reduction = ((avg_words_before - avg_words_after) / avg_words_before) * 100
        st.metric("Reduksi teks", f"{reduction:.1f}%")

# ==================== HALAMAN ANALISIS MODEL ====================
def show_model_analysis(df, model_data):
    """Menampilkan analisis model"""
    st.header("📈 Analisis Model")
    
    # Evaluasi model
    st.subheader("📊 Evaluasi Performa Model")
    
    # Cross-validation dengan Leave-One-Out
    with st.spinner("Menghitung evaluasi model..."):
        loo = LeaveOneOut()
        cv_scores = cross_val_score(
            model_data['model'], 
            model_data['X'], 
            model_data['y'], 
            cv=loo, 
            scoring='accuracy'
        )
    
    # Tampilkan metrik
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi Rata-rata", f"{cv_scores.mean():.2%}")
    
    with col2:
        st.metric("Akurasi Tertinggi", f"{cv_scores.max():.2%}")
    
    with col3:
        st.metric("Akurasi Terendah", f"{cv_scores.min():.2%}")
    
    with col4:
        st.metric("Standar Deviasi", f"{cv_scores.std():.4f}")
    
    # Confusion matrix
    st.subheader("🎯 Confusion Matrix")
    
    # Prediksi untuk seluruh data
    y_pred = model_data['model'].predict(model_data['X'])
    
    # Buat confusion matrix
    labels = list(model_data['label_mapping'].keys())
    cm = confusion_matrix(model_data['y'], y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Jumlah Kasus'})
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    st.pyplot(fig)
    
    # Classification report
    st.subheader("📋 Classification Report")
    
    y_true_labels = [model_data['reverse_mapping'][label] for label in model_data['y']]
    y_pred_labels = [model_data['reverse_mapping'][label] for label in y_pred]
    
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.dataframe(report_df, use_container_width=True, height=400)
    
    # Analisis fitur penting
    st.subheader("🔍 Analisis Fitur Penting (TF-IDF)")
    
    selected_class = st.selectbox(
        "Pilih jenis perkara untuk analisis fitur:",
        list(model_data['label_mapping'].keys())
    )
    
    class_idx = model_data['label_mapping'][selected_class]
    feature_names = model_data['vectorizer'].get_feature_names_out()
    feature_log_probs = model_data['model'].feature_log_prob_[class_idx]
    
    # Slider untuk jumlah fitur
    top_n = st.slider("Jumlah fitur yang ditampilkan:", 5, 25, 15)
    
    # Ambil top dan bottom features
    top_indices = np.argsort(feature_log_probs)[-top_n:][::-1]
    bottom_indices = np.argsort(feature_log_probs)[:top_n]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Top {top_n} fitur untuk '{selected_class}':**")
        top_features = []
        for idx in top_indices:
            if feature_log_probs[idx] > -10:
                top_features.append({
                    'Fitur': feature_names[idx],
                    'Bobot': feature_log_probs[idx]
                })
        top_df = pd.DataFrame(top_features)
        st.dataframe(top_df, use_container_width=True, height=400)
    
    with col2:
        st.write(f"**Bottom {top_n} fitur untuk '{selected_class}':**")
        bottom_features = []
        for idx in bottom_indices:
            if feature_log_probs[idx] < 0:
                bottom_features.append({
                    'Fitur': feature_names[idx],
                    'Bobot': feature_log_probs[idx]
                })
        bottom_df = pd.DataFrame(bottom_features)
        st.dataframe(bottom_df, use_container_width=True, height=400)
    
    # Visualisasi fitur
    st.subheader("📊 Visualisasi Bobot Fitur")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Top features visualization
    if len(top_features) > 0:
        top_values = [f['Bobot'] for f in top_features]
        top_labels = [f['Fitur'] for f in top_features]
        
        bars1 = ax1.barh(range(len(top_values)), top_values, color='lightgreen')
        ax1.set_yticks(range(len(top_values)))
        ax1.set_yticklabels(top_labels)
        ax1.invert_yaxis()
        ax1.set_xlabel('Bobot (Log Probability)')
        ax1.set_title(f'Top {len(top_features)} Fitur untuk "{selected_class}"')
        
        # Tambah nilai pada bar
        for bar, value in zip(bars1, top_values):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center')
    
    # Bottom features visualization
    if len(bottom_features) > 0:
        bottom_values = [f['Bobot'] for f in bottom_features]
        bottom_labels = [f['Fitur'] for f in bottom_features]
        
        bars2 = ax2.barh(range(len(bottom_values)), bottom_values, color='lightcoral')
        ax2.set_yticks(range(len(bottom_values)))
        ax2.set_yticklabels(bottom_labels)
        ax2.invert_yaxis()
        ax2.set_xlabel('Bobot (Log Probability)')
        ax2.set_title(f'Bottom {len(bottom_features)} Fitur untuk "{selected_class}"')
        
        # Tambah nilai pada bar
        for bar, value in zip(bars2, bottom_values):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Informasi model
    st.subheader("ℹ️ Informasi Teknis Model")
    
    with st.expander("Lihat detail model"):
        st.write(f"**Parameter TF-IDF:**")
        st.write(f"- Max Features: {model_data['vectorizer'].max_features}")
        st.write(f"- N-gram Range: {model_data['vectorizer'].ngram_range}")
        st.write(f"- Vocabulary Size: {len(model_data['vectorizer'].vocabulary_)}")
        
        st.write(f"\n**Parameter Naive Bayes:**")
        st.write(f"- Alpha (smoothing): {model_data['model'].alpha}")
        st.write(f"- Number of Classes: {model_data['model'].n_classes_}")
        st.write(f"- Feature Count per Class: {model_data['model'].feature_count_.shape}")

# ==================== HALAMAN PANDUAN ====================
def show_guide():
    """Menampilkan halaman panduan"""
    st.header("📘 Panduan Penggunaan Sistem")
    
    st.markdown("""
    ## 🎯 **Tujuan Sistem**
    Sistem ini dirancang untuk mengklasifikasikan jenis perkara kriminal berdasarkan deskripsi singkat kejadian menggunakan metode **Naive Bayes** dan **TF-IDF** dengan preprocessing **Sastrawi** untuk Bahasa Indonesia.
    
    ## 📋 **Menu Navigasi**
    
    ### 1. 📊 **Dashboard**
    - Menampilkan statistik dan visualisasi data kasus kriminal
    - Distribusi jenis perkara
    - Lokasi kejadian paling banyak
    
    ### 2. 🔮 **Prediksi Kasus**
    - Masukkan deskripsi kasus kriminal
- Lihat hasil klasifikasi dengan tingkat keyakinan
- Analisis kata kunci yang mempengaruhi prediksi
- Riwayat prediksi sebelumnya
    
    ### 3. 📚 **Data & Preprocessing**
    - Lihat data asli dan hasil preprocessing
    - Demo proses preprocessing teks
    - Statistik sebelum dan sesudah preprocessing
    
    ### 4. 📈 **Analisis Model**
    - Evaluasi performa model (akurasi, confusion matrix)
    - Analisis fitur penting (TF-IDF)
    - Informasi teknis model
    
    ## 🔧 **Cara Menggunakan**
    
    ### **Langkah 1: Install Dependencies**
    Pastikan semua package terinstal dengan menekan tombol **"Install Packages"** di sidebar jika ada package yang belum terinstal.
    
    ### **Langkah 2: Upload Data (Opsional)**
    - Upload file Excel Anda melalui sidebar
    - File harus memiliki kolom: **'URAIAN SINGKAT (MO)'** dan **'PERKARA'**
    - Jika tidak upload, sistem akan menggunakan data contoh
    
    ### **Langkah 3: Konfigurasi**
    - Pilih apakah ingin menggunakan **Sastrawi** untuk preprocessing
    - Sastrawi akan melakukan **stemming** dan **stopword removal** Bahasa Indonesia
    
    ### **Langkah 4: Prediksi Kasus**
    1. Pilih menu **"Prediksi Kasus"**
    2. Masukkan deskripsi kasus atau pilih contoh
    3. Klik **"Analisis Kasus"**
    4. Lihat hasil klasifikasi dan analisis
    
    ## 📊 **Metodologi CRISP-DM**
    
    Sistem ini mengimplementasikan metodologi **CRISP-DM** (Cross Industry Standard Process for Data Mining):
    
    1. **Business Understanding**: Memahami kebutuhan klasifikasi kasus kriminal
    2. **Data Understanding**: Eksplorasi dan analisis data kasus
    3. **Data Preparation**: Preprocessing teks dengan Sastrawi
    4. **Modeling**: Implementasi Naive Bayes dengan TF-IDF
    5. **Evaluation**: Evaluasi model dengan cross-validation
    6. **Deployment**: Sistem web interaktif dengan Streamlit
    
    ## 🔍 **Algoritma yang Digunakan**
    
    ### **1. TF-IDF (Term Frequency-Inverse Document Frequency)**
    - Mengubah teks menjadi vektor numerik
    - Memberi bobot pada kata berdasarkan frekuensi dan kelangkaan
    
    ### **2. Naive Bayes Multinomial**
    - Algoritma klasifikasi probabilistik
    - Cocok untuk data teks dengan banyak fitur
    - Efisien untuk dataset kecil hingga menengah
    
    ### **3. Sastrawi untuk Preprocessing**
    - **Stemming**: Mengubah kata ke bentuk dasar
    - **Stopword Removal**: Menghapus kata umum yang tidak informatif
    
    ## 💡 **Tips untuk Hasil Terbaik**
    
    1. **Deskripsi yang Jelas**: Tulis deskripsi kasus dengan detail yang cukup
    2. **Bahasa Indonesia Formal**: Gunakan bahasa Indonesia yang baik dan benar
    3. **Fokus pada Fakta**: Deskripsikan kejadian tanpa opini atau emosi
    4. **Konsistensi**: Gunakan istilah yang konsisten dalam deskripsi
    
    ## 🛠️ **Troubleshooting**
    
    ### **Problem: Package tidak terinstal**
    **Solution**: Klik tombol "Install Packages" di sidebar
    
    ### **Problem: File Excel tidak terbaca**
    **Solution**: 
    - Pastikan format file .xlsx atau .xls
    - Pastikan kolom yang diperlukan ada
    - Coba gunakan data contoh terlebih dahulu
    
    ### **Problem: Prediksi tidak akurat**
    **Solution**:
    - Tambah data training
    - Gunakan deskripsi yang lebih detail
    - Coba dengan dan tanpa Sastrawi
    
    ## 📞 **Dukungan**
    
    Sistem ini dikembangkan untuk penelitian **Klasifikasi Kasus Kriminal**.
    Untuk pertanyaan atau masalah, silakan hubungi developer.
    """)

# ==================== RUN APLIKASI ====================
if __name__ == "__main__":
    main()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Sistem Klasifikasi Kasus Kriminal**
    
    **Algoritma**: Naive Bayes dengan TF-IDF  
    **Preprocessing**: Sastrawi (Stemming & Stopword Removal)  
    **Metodologi**: CRISP-DM  
    
    © 2024 - Sistem Klasifikasi Kriminal
    """)