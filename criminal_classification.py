import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class CriminalCaseClassifier:
    def __init__(self):
        """Initialize the classifier with Indonesian language processing"""
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        
        # Indonesian stopwords
        self.stop_words = set(stopwords.words('indonesian'))
        # Add custom stopwords for criminal cases
        custom_stopwords = ['pelaku', 'korban', 'melakukan', 'dengan', 'yang', 'di', 'ke', 'dari', 'dan', 'atau']
        self.stop_words.update(custom_stopwords)
        
        self.vectorizer = None
        self.model = None
        self.classes = None
        
    def load_data(self, excel_path='Data.xlsx'):
        """Load data from Excel file"""
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
        print(f"Data loaded: {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Class distribution:\n{df['PERKARA'].value_counts()}")
        return df
    
    def preprocess_text(self, text):
        """Preprocess Indonesian text with multiple steps"""
        if pd.isna(text):
            return ""
        
        # 1. Case folding
        text = text.lower()
        
        # 2. Remove special characters and numbers
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Stopword removal
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # 5. Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        # 6. Remove short words
        tokens = [word for word in tokens if len(word) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, df):
        """Extract features from text descriptions"""
        # Preprocess all text
        df['cleaned_text'] = df['URAIAN SINGKAT (MO)'].apply(self.preprocess_text)
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=1,
            max_df=0.8,
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['PERKARA']
        
        self.classes = y.unique()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Classes: {self.classes}")
        
        return X, y, df
    
    def train_model(self, X, y):
        """Train Naive Bayes model with hyperparameter tuning"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('classifier', MultinomialNB())
        ])
        
        # Define parameters for grid search
        parameters = {
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'classifier__fit_prior': [True, False]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, 
            parameters, 
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        
        print("Training model with grid search...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
        
        # Test set evaluation
        y_pred = self.model.predict(X_test)
        print("\nTest Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return grid_search.best_estimator_
    
    def visualize_data(self, df):
        """Create visualizations for data understanding"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Class distribution
        ax1 = axes[0, 0]
        class_counts = df['PERKARA'].value_counts()
        ax1.bar(class_counts.index, class_counts.values)
        ax1.set_title('Distribusi Kasus Kriminal')
        ax1.set_xlabel('Jenis Perkara')
        ax1.set_ylabel('Jumlah Kasus')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Word cloud
        ax2 = axes[0, 1]
        all_text = ' '.join(df['cleaned_text'])
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=50
        ).generate(all_text)
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.set_title('Word Cloud dari Deskripsi Kasus')
        ax2.axis('off')
        
        # 3. Top words per class
        ax3 = axes[1, 0]
        top_words = {}
        for crime_type in df['PERKARA'].unique():
            text = ' '.join(df[df['PERKARA'] == crime_type]['cleaned_text'])
            words = text.split()
            word_counts = Counter(words).most_common(10)
            top_words[crime_type] = [w[0] for w in word_counts]
        
        # Create a simple bar chart for one class (Pencurian has most data)
        if 'Pencurian' in top_words:
            words = top_words['Pencurian'][:5]
            counts = [all_text.split().count(w) for w in words]
            ax3.bar(words, counts)
            ax3.set_title('Kata-kata Umum dalam Kasus Pencurian')
            ax3.set_xlabel('Kata')
            ax3.set_ylabel('Frekuensi')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Text length distribution
        ax4 = axes[1, 1]
        df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        ax4.hist(df['text_length'], bins=10, edgecolor='black')
        ax4.set_title('Distribusi Panjang Teks')
        ax4.set_xlabel('Jumlah Kata')
        ax4.set_ylabel('Frekuensi')
        
        plt.tight_layout()
        plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def predict_new_case(self, description):
        """Predict class for a new case description"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model belum dilatih. Train model terlebih dahulu.")
        
        # Preprocess the new description
        cleaned_text = self.preprocess_text(description)
        
        # Transform using the same vectorizer
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)
        probabilities = self.model.predict_proba(text_vectorized)[0]
        
        result = {
            'prediction': prediction[0],
            'probabilities': dict(zip(self.classes, probabilities)),
            'cleaned_text': cleaned_text
        }
        
        return result
    
    def save_model(self, filename='criminal_classifier.pkl'):
        """Save the trained model and vectorizer"""
        import joblib
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'classes': self.classes,
            'stop_words': self.stop_words
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='criminal_classifier.pkl'):
        """Load a trained model"""
        import joblib
        
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.classes = model_data['classes']
        self.stop_words = model_data['stop_words']
        
        print(f"Model loaded from {filename}")
        print(f"Classes: {self.classes}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("KLASIFIKASI KASUS KRIMINAL - METODE NAIVE BAYES & TF-IDF")
    print("=" * 60)
    
    # Initialize classifier
    classifier = CriminalCaseClassifier()
    
    # Load data
    df = classifier.load_data('Data.xlsx')
    
    # Extract features
    X, y, df = classifier.extract_features(df)
    
    # Train model
    model = classifier.train_model(X, y)
    
    # Visualize data
    print("\n" + "=" * 60)
    print("VISUALISASI DATA")
    print("=" * 60)
    classifier.visualize_data(df)
    
    # Save model
    classifier.save_model()
    
    # Example predictions
    print("\n" + "=" * 60)
    print("CONTOH PREDIKSI")
    print("=" * 60)
    
    test_cases = [
        "pelaku mengambil handphone dari meja dengan paksa",
        "terdakwa memukul korban hingga luka berat",
        "pencurian mobil dengan cara membobol kunci",
        "pengeroyokan oleh sekelompok orang terhadap satu korban"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = classifier.predict_new_case(test_case)
        print(f"\nTest case {i}:")
        print(f"Input: {test_case}")
        print(f"Cleaned: {result['cleaned_text'][:100]}...")
        print(f"Prediksi: {result['prediction']}")
        print("Probabilitas:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("PROSES SELESAI")
    print("=" * 60)

if __name__ == "__main__":
    main()