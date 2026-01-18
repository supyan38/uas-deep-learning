
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. INPUT DATA TRAINING (Simulasi Dataset)
# ==========================================
data = {
    'text': [
        "Debat capres kali ini sangat panas dan argumennya kuat.",
        "Saya tidak suka dengan cara penyampaian salah satu calon, terkesan emosi.",
        "Data yang disampaikan Pak Anies sangat akurat dan membuka mata.",
        "Prabowo kurang tenang dalam debat ini, harusnya lebih sabar.",
        "Ganjar tampil memukau dengan program kerjanya yang realistis.",
        "Debat isinya saling serang, tidak produktif.",
        "Sangat kecewa dengan kualitas debat semalam.",
        "Hebat, gagasan tentang pertahanan negara sangat visioner!",
        "Tidak ada substansi, hanya omong kosong.",
        "Saya makin yakin memilih paslon nomor urut tertentu setelah nonton debat.",
        "Lucu sekali melihat meme debat yang beredar.",
        "Suasana debat tegang tapi seru untuk ditonton.",
        "capres harusnya fokus solusi bukan menyerang personal",
        "bangga dengan demokrasi indonesia",
        "sangat membosankan debatnya"
    ],
    'label': [
        'Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 
        'Negatif', 'Negatif', 'Positif', 'Negatif', 'Positif', 
        'Positif', 'Positif', 'Negatif', 'Positif', 'Negatif'
    ]
}

df = pd.DataFrame(data)

# ==========================================
# 2. PREPROCESSING
# ==========================================
def clean_text(text):
    # Case folding
    text = text.lower()
    # Remove special chars & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (simple split)
    tokens = text.split()
    # Filtering / Stopword removal (Manual list for simulation)
    stopwords = {'dan', 'yang', 'di', 'ini', 'itu', 'dengan', 'dari', 'saya', 'tidak', 'tapi', 'ke'}
    tokens = [word for word in tokens if word not in stopwords]
    # Stemming (Simplified without Sastrawi to ensure runnability, usually use Sastrawi here)
    # In real heavy implementation we would use Sastrawi.
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

print("Data setelah preprocessing:")
print(df[['text', 'clean_text']].head())

# ==========================================
# 3. MODELLING & 4. EVALUATION
# ==========================================
# Split Data
X = df['clean_text']
y = df['label']

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# 5. PREDICTION (Save Model & Vectorizer)
# ==========================================
# Saving properly for the Streamlit App
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel dan Vectorizer telah disimpan.")
