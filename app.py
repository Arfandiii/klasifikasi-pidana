import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Klasifikasi Kasus Kriminal",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498DB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
        margin: 20px 0;
    }
    .probability-bar {
        height: 20px;
        background-color: #3498DB;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class CriminalClassificationApp:
    def __init__(self):
        self.classifier = None
        self.model_data = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model_data = joblib.load('criminal_classifier.pkl')
            return True
        except:
            return False
    
    def predict(self, text):
        """Make prediction for input text"""
        # Preprocess text
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stop_words = set(stopwords.words('indonesian'))
        
        # Preprocessing function
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [stemmer.stem(word) for word in tokens]
            tokens = [word for word in tokens if len(word) > 2]
            return ' '.join(tokens)
        
        cleaned_text = preprocess_text(text)
        vectorized = self.model_data['vectorizer'].transform([cleaned_text])
        prediction = self.model_data['model'].predict(vectorized)[0]
        probabilities = self.model_data['model'].predict_proba(vectorized)[0]
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.model_data['classes'], probabilities)),
            'cleaned_text': cleaned_text
        }

def main():
    app = CriminalClassificationApp()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/police-badge.png", width=100)
        st.title("Navigasi")
        
        menu = st.radio(
            "Menu",
            ["🏠 Dashboard", "🔍 Klasifikasi Kasus", "📊 Analisis Data", "ℹ️ Tentang"]
        )
        
        st.markdown("---")
        st.info(
            "Aplikasi ini menggunakan metode Naive Bayes dan TF-IDF "
            "untuk klasifikasi kasus kriminal berdasarkan deskripsi kejadian."
        )
    
    # Main content
    if menu == "🏠 Dashboard":
        st.markdown('<h1 class="main-header">Sistem Klasifikasi Kasus Kriminal</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Metode", "Naive Bayes", delta="Algoritma")
        
        with col2:
            st.metric("Fitur", "TF-IDF", delta="Pembobotan")
        
        with col3:
            st.metric("Akurasi", "> 85%", delta="Estimasi")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Fitur Utama</h3>', unsafe_allow_html=True)
            st.markdown("""
            ✅ **Klasifikasi Otomatis**  
            ✅ **Analisis Probabilitas**  
            ✅ **Visualisasi Data**  
            ✅ **Preprocessing Teks**  
            ✅ **Stemming Bahasa Indonesia**  
            ✅ **Stopword Removal**  
            """)
        
        with col2:
            st.markdown('<h3 class="sub-header">Jenis Kasus</h3>', unsafe_allow_html=True)
            st.markdown("""
            🔹 **Pencurian**  
            🔹 **Penganiayaan**  
            🔹 **Pencurian dengan Kekerasan**  
            🔹 **Penganiayaan Ringan**  
            """)
    
    elif menu == "🔍 Klasifikasi Kasus":
        st.markdown('<h1 class="main-header">Klasifikasi Kasus Baru</h1>', unsafe_allow_html=True)
        
        if not app.load_model():
            st.error("Model belum tersedia. Silakan train model terlebih dahulu.")
            return
        
        # Input form
        with st.form("classification_form"):
            st.markdown("### Masukkan Deskripsi Kasus")
            
            case_description = st.text_area(
                "Deskripsi Kejadian:",
                height=150,
                placeholder="Contoh: Pelaku memasuki rumah dengan membuka kunci dan mengambil handphone dari meja..."
            )
            
            submitted = st.form_submit_button("🔍 Analisis Kasus")
        
        if submitted and case_description:
            with st.spinner("Menganalisis deskripsi kasus..."):
                result = app.predict(case_description)
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Prediksi: **{result['prediction']}**")
                
                # Display probability bars
                st.markdown("#### Probabilitas:")
                for crime_type, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    percentage = prob * 100
                    st.markdown(f"**{crime_type}**")
                    st.progress(float(prob))
                    st.markdown(f"{percentage:.1f}%")
                    st.markdown("---")
            
            with col2:
                st.markdown("#### Teks yang Diproses:")
                st.info(result['cleaned_text'])
                
                st.markdown("#### Statistik:")
                word_count = len(result['cleaned_text'].split())
                char_count = len(result['cleaned_text'])
                st.metric("Jumlah Kata", word_count)
                st.metric("Jumlah Karakter", char_count)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "📊 Analisis Data":
        st.markdown('<h1 class="main-header">Analisis Data Kasus</h1>', unsafe_allow_html=True)
        
        # Load and display data
        try:
            df = pd.read_excel('Data.xlsx')
            
            tab1, tab2, tab3 = st.tabs(["📈 Dataset", "📊 Visualisasi", "🔠 Kata-kata Kunci"])
            
            with tab1:
                st.dataframe(df, use_container_width=True)
                
                st.markdown("### Statistik Dataset")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Kasus", len(df))
                with col2:
                    st.metric("Jenis Kasus", df['PERKARA'].nunique())
                with col3:
                    st.metric("Kolom Fitur", len(df.columns))
            
            with tab2:
                st.markdown("### Distribusi Kasus Kriminal")
                
                # Class distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                class_counts = df['PERKARA'].value_counts()
                bars = ax.bar(class_counts.index, class_counts.values, color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'])
                ax.set_xlabel('Jenis Perkara')
                ax.set_ylabel('Jumlah Kasus')
                ax.set_title('Distribusi Kasus Kriminal')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with tab3:
                st.markdown("### Kata-kata yang Sering Muncul")
                
                # Create word cloud from descriptions
                all_text = ' '.join(df['URAIAN SINGKAT (MO)'].dropna().str.lower())
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=50,
                    colormap='viridis'
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud dari Deskripsi Kasus')
                
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Tidak dapat memuat data: {str(e)}")
    
    elif menu == "ℹ️ Tentang":
        st.markdown('<h1 class="main-header">Tentang Aplikasi</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Tujuan
        Aplikasi ini dikembangkan untuk membantu dalam klasifikasi otomatis kasus kriminal 
        berdasarkan deskripsi kejadian menggunakan teknik Natural Language Processing (NLP).
        
        ### 🔧 Teknologi yang Digunakan
        - **Python** sebagai bahasa pemrograman utama
        - **Streamlit** untuk antarmuka web
        - **Scikit-learn** untuk machine learning
        - **NLTK & Sastrawi** untuk preprocessing teks Bahasa Indonesia
        - **Naive Bayes** sebagai algoritma klasifikasi
        - **TF-IDF** untuk pembobotan fitur teks
        
        ### 📚 Metodologi CRISP-DM
        Aplikasi ini mengikuti metodologi CRISP-DM:
        
        1. **Business Understanding** - Memahami kebutuhan klasifikasi kasus
        2. **Data Understanding** - Analisis dataset kasus kriminal
        3. **Data Preparation** - Preprocessing teks dan ekstraksi fitur
        4. **Modeling** - Pelatihan model Naive Bayes
        5. **Evaluation** - Evaluasi performa model
        6. **Deployment** - Implementasi dalam aplikasi web
        
        ### 👨‍💻 Pengembang
        Sistem ini dikembangkan untuk penelitian klasifikasi kasus kriminal 
        menggunakan metode machine learning.
        
        ### 📄 Lisensi
        Aplikasi ini dikembangkan untuk tujuan akademis dan penelitian.
        """)

if __name__ == "__main__":
    main()