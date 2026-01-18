
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss

# Try to import Sastrawi, provide fallback if missing
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# Try to import NLTK
import nltk
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# ==========================================
# CUSTOM CSS & CONFIG
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen Debat Capres 2024",
    page_icon="🗳️",
    layout="wide"
)

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #F8FAFC; /* Sangat light grey-blue */
    }

    /* Header Styles */
    .main-header {
        text-align: center;
        color: #4F46E5; /* Indigo */
        font-weight: 700;
        padding: 2rem 0;
        text-shadow: 1px 1px 2px rgba(79, 70, 229, 0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
    }

    /* Tab Customization (Streamlit default tabs are tricky, adding wrapper style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748B;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F46E5;
        color: white;
    }

    /* Card Styling */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border-left: 5px solid #4F46E5;
    }

    /* Steps / Process Titles */
    .step-title {
        color: #1E293B;
        font-weight: 600;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        height: 3.5em; 
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.23);
        color: white;
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #F1F5F9 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        width: 100%;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F46E5;
    }
    .metric-label {
        color: #64748B;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Prediction Result Box */
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s;
    }
    .pred-pos {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .pred-neg {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCTIONS WITH PREFIX 'npm_20221310078'
# ==========================================

@st.cache_data
def npm_20221310078_load_data():
    """Load data training"""
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/rasyidev/well-known-datasets/main/juli2train.csv')
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

def npm_20221310078_get_stopwords():
    """Get stopword list (Indonesian)"""
    stops = {'dan', 'yang', 'di', 'ini', 'itu', 'dengan', 'dari', 'saya', 'tidak', 'tapi', 'ke', 'ada', 'adalah', 'akan'}
    if NLTK_AVAILABLE:
        try:
            stops.update(nltk_stopwords.words('indonesian'))
        except:
            pass
    try:
        with open('stopwordlist.txt', 'r') as f:
            file_stops = f.read().split()
            stops.update(file_stops)
    except FileNotFoundError:
        pass
    return list(stops)

def npm_20221310078_stemmer():
    """Create stemmer instance"""
    if SASTRAWI_AVAILABLE:
        factory = StemmerFactory()
        return factory.create_stemmer()
    else:
        return None

def npm_20221310078_preprocess_text(text, stop_words, stemmer_obj):
    """Cleaning, Stopword Removal, Stemming"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            if stemmer_obj:
                try:
                    token = stemmer_obj.stem(token)
                except:
                    pass
            clean_tokens.append(token)
    return " ".join(clean_tokens)

@st.cache_data
def npm_20221310078_run_preprocessing(df):
    """Run full preprocessing pipeline on dataframe"""
    stop_words = npm_20221310078_get_stopwords()
    stemmer_obj = npm_20221310078_stemmer()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    clean_texts = []
    total = len(df)
    
    for i, row in df.iterrows():
        clean = npm_20221310078_preprocess_text(row['tweet'], stop_words, stemmer_obj)
        clean_texts.append(clean)
        if (i + 1) % 50 == 0:
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing... {int(progress*100)}%")
            
    progress_bar.progress(1.0)
    status_text.empty()
    df['clean_twt'] = clean_texts
    return df

@st.cache_resource
def npm_20221310078_train_model(df):
    """Train Tfidf and SVM/Logistic Regression"""
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=10)
    features = vectorizer.fit_transform(df['clean_twt'])
    X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.2, random_state=4)
    model = LogisticRegression(C=3, solver='liblinear', max_iter=150)
    model.fit(X_train, y_train)
    return model, vectorizer, X_test, y_test

# ==========================================
# MAIN INTERFACE
# ==========================================

def main():
    st.markdown("<h1 class='main-header'>🗳️ Analisis Sentimen Debat Capres</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Implementasi Logistic Regression untuk Klasifikasi Sentimen</div>", unsafe_allow_html=True)
    
    # Create Tabs with cleaner names
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Data Input", 
        "⚙️ Preprocessing", 
        "🧠 Modeling", 
        "📊 Evaluasi", 
        "🔮 Prediksi"
    ])
    
    # Session State Init
    if 'data_raw' not in st.session_state: st.session_state['data_raw'] = None
    if 'data_clean' not in st.session_state: st.session_state['data_clean'] = None
    if 'model_obj' not in st.session_state: st.session_state['model_obj'] = None
    
    # TAB 1: INPUT DATA
    with tab1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='step-title'>Proses 1: Input Data Training & Crawling</h3>", unsafe_allow_html=True)
        st.write("Mengambil data training dataset `juli2train.csv`.")
        st.caption("Sumber Data: Kompas.id (Penguasaan Data dalam Debat Capres)")
        
        if st.button("🚀 Load Data Training", key='btn_load'):
            with st.spinner("Mengunduh dataset..."):
                df = npm_20221310078_load_data()
                if df is not None:
                    st.session_state['data_raw'] = df
                    st.success(f"Berhasil memuat {len(df)} data.")
        
        if st.session_state['data_raw'] is not None:
            df = st.session_state['data_raw']
            st.divider()
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write("**Preview Data:**")
                st.dataframe(df.head(5), use_container_width=True)
            with col_b:
                st.write("**Distribusi Label:**")
                st.bar_chart(df['label'].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 2: PREPROCESSING
    with tab2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='step-title'>Proses 2: Text Preprocessing</h3>", unsafe_allow_html=True)
        st.info("💡 Tahapan: Cleaning ➔ Case Folding ➔ Stopword Removal ➔ Stemming")
        
        if not SASTRAWI_AVAILABLE:
            st.warning("⚠️ Library Sastrawi tidak ditemukan. Stemming skips.")

        if st.session_state['data_raw'] is not None:
            if st.button("⚡ Mulai Preprocessing", key='btn_prepro'):
                df_clean = npm_20221310078_run_preprocessing(st.session_state['data_raw'].copy())
                st.session_state['data_clean'] = df_clean
                st.success("Preprocessing selesai!")
        else:
            st.warning("Mohon Load Data di Tab 1 dahulu.")
            
        if st.session_state['data_clean'] is not None:
            st.divider()
            st.write("**Hasil Preprocessing:**")
            st.dataframe(st.session_state['data_clean'][['tweet', 'clean_twt']].head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3: MODELING
    with tab3:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='step-title'>Proses 3: Modeling (Logistic Regression)</h3>", unsafe_allow_html=True)
        st.write("Melatih model machine learning menggunakan fitur TF-IDF.")
        
        if st.session_state['data_clean'] is not None:
            if st.button("🛠️ Latih Model", key='btn_train'):
                with st.spinner("Training model sedang berjalan..."):
                    model, vectorizer, X_test, y_test = npm_20221310078_train_model(st.session_state['data_clean'])
                    st.session_state['model_obj'] = model
                    st.session_state['vectorizer_obj'] = vectorizer
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.success("Model berhasil dilatih!")
        else:
            st.info("Input Data & Preprocessing belum selesai.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 4: EVALUATION
    with tab4:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='step-title'>Proses 4: Evaluasi Model</h3>", unsafe_allow_html=True)
        
        if st.session_state['model_obj'] is not None:
            model = st.session_state['model_obj']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            yhat = model.predict(X_test)
            yhat_prob = model.predict_proba(X_test)
            f1 = f1_score(y_test, yhat, average='weighted')
            loss = log_loss(y_test, yhat_prob)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{f1:.2%}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{loss:.4f}</div>
                    <div class="metric-label">Log Loss</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model belum dilatih. Silakan ke Tab 3.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 5: PREDICTION
    with tab5:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='step-title'>Proses 5: Uji Coba Prediksi</h3>", unsafe_allow_html=True)
        
        mode = st.radio("Pilih Model:", ["Model Baru (Sesi Ini)", "Model Tersimpan (Pickle)"], horizontal=True)
        
        mod, vec = None, None
        if mode == "Model Baru (Sesi Ini)":
            if st.session_state['model_obj']:
                mod, vec = st.session_state['model_obj'], st.session_state['vectorizer_obj']
            else: st.warning("Belum ada model baru.")
        else:
            try:
                with open('sentiment_model.pkl', 'rb') as f: mod = pickle.load(f)
                with open('tfidf_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
            except: st.error("File model pickle tidak ditemukan.")

        input_text = st.text_area("💬 Masukkan Komentar Anda:", height=100)
        
        if st.button("🔍 Analisis Sentimen", key='btn_predict'):
            if input_text and mod and vec:
                clean_txt = npm_20221310078_preprocess_text(input_text, npm_20221310078_get_stopwords(), npm_20221310078_stemmer())
                pred = mod.predict(vec.transform([clean_txt]))[0]
                
                label = "POSITIF" if str(pred) == '1' or pred == 1 else "NEGATIF"
                css_class = "pred-pos" if label == "POSITIF" else "pred-neg"
                icon = "😊" if label == "POSITIF" else "😡"
                
                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    <h2>{icon} {label}</h2>
                    <p>"{input_text}"</p>
                </div>
                """, unsafe_allow_html=True)
            elif not input_text:
                st.warning("Teks tidak boleh kosong.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
