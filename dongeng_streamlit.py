import streamlit as st
st.set_page_config(
    page_title="Dongeng Nusantara Panji Kediri",
    page_icon="📚"
)
import joblib
import numpy as np
import os
import re
import csv
from gensim.matutils import sparse2full
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Styling function
def add_styles():
    st.markdown(
        """
        <style>
         .title-box {
            background-color: #FFFFFF; 
            padding: 20px;           
            border-radius: 10px;      
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); 
            text-align: center;       
            color: #000000;            
            font-size: 30px;          
            font-weight: bold;       
            font-family: 'Arial', sans-serif; 
        }
        .subtitle {
            font-size: 15px;          
            color: #000000;          
            margin-top: 10px;         
            font-weight: normal;       
        }
        .stApp {
            background: #e6f0ff;
            font-family: Arial, sans-serif;
        }
        .search-container {
            margin-top: 10px;
            margin-bottom: 10px;
        }
         .search-label {
            color: #1a3e72;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 2px !important;
        }
        .result-box {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #1a3e72;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: -10px;
        }
        .result-title {
            color: #1a3e72;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .result-preview {
            color: #333333;
            font-size: 1.0em;
            line-height: 1.6;
            margin-bottom: 15px;
            text-align: justify;
        }
        .result-info {
            color: #666666;
            font-size: 0.85em;
            border-top: 1px solid #e0e0e0;
            padding-top: 8px;
            margin-top: 10px;
        }
        
        /* Styling untuk highlight kata kunci */
        .highlight-keyword {
            background-color: #1a3e72 !important;
            color: white !important;
            font-weight: bold !important;
            padding: 2px 4px !important;
            border-radius: 3px !important;
            text-decoration: none !important;
        }
        
        /* Updated Alert/Warning styling to match button colors */
        .stAlert {
            background-color: #1a3e72 !important;
            border: 1px solid #1a3e72 !important;
            border-radius: 8px !important;
            border-left: 4px solid #0d2b57 !important;
            color: white !important;
        }
        
        .stAlert > div {
            color: white !important;
        }
        
        .stAlert svg {
            fill: white !important;
        }
        
        .stAlert:hover {
            background-color: #0d2b57 !important;
            border-color: #0d2b57 !important;
        }
        
        /* Specific styling for warning alerts */
        div[data-testid="stAlert"] {
            background-color: #1a3e72 !important;
            border: 1px solid #1a3e72 !important;
            color: white !important;
        }
        
        div[data-testid="stAlert"] > div {
            color: white !important;
        }
        
        /* Alternative selector for alerts */
        .element-container .stAlert {
            background-color: #1a3e72 !important;
            border-color: #1a3e72 !important;
            color: white !important;
        }
        
        button[kind="primary"], button[kind="secondary"] {
            background-color: #1a3e72 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin-top: -10px;
        }
        button[kind="primary"]:hover, button[kind="secondary"]:hover {
            background-color: #0d2b57 !important;
        }
        .stTextInput>div>div>input {
            background-color: white !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            width: 100% !important;
        }
        
        /* Placeholder text */
        .stTextInput>div>div>input::placeholder {
            color: #6c757d !important;
            opacity: 1 !important;
        }
        
        /* Cursor color */
        .stTextInput>div>div>input {
            caret-color: black !important;
        }
        
        /* Hover state */
        .stTextInput>div>div>input:hover {
            border-color: #1a3e72 !important;
        }
        
        /* Focus state */
        .stTextInput>div>div>input:focus {
            border-color: #1a3e72 !important;
            box-shadow: 0 0 0 2px rgba(26, 62, 114, 0.2) !important;
        }
        
        /* Style untuk button baca lengkap */
        .read-button {
            display: block !important;
            width: 100% !important;
            background-color: #28a745 !important;
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 8px 16px !important;
            font-size: 0.9em !important;
            margin-top: 0px !important;
            text-align: center !important;
        }
        .read-button:hover {
            background-color: #218838 !important;
        }
        
        /* Style untuk modal/popup */
        .modal-content {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: -50px;
            overflow-y: auto;
            border: 2px solid #1a3e72;
        }
        .modal-title {
            color: #1a3e72;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            border-bottom: 2px solid #1a3e72;
            padding-bottom: 10px;
        }
       .modal-story {
    max-height: 60vh;
    overflow-y: auto;
    
    scrollbar-width: thin;  /* Untuk browser modern */
}

/* Scrollbar untuk WebKit (Chrome, Safari) */
.modal-story::-webkit-scrollbar {
    width: 8px;
}
.modal-story::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}
.modal-story::-webkit-scrollbar-thumb {
    background: #1a3e72;
    border-radius: 10px;
} .modal-story {
            color: #333333;
            font-size: 1.0em;
            line-height: 1.8;
            text-align: justify;
            margin-top: -10px;
            max-height: 60vh;
            overflow-y: auto;
            white-space: pre-line;
        }
        
        /* Style scrollbar untuk modal */
        .modal-story::-webkit-scrollbar {
            width: 8px;
        }
        .modal-story::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .modal-story::-webkit-scrollbar-thumb {
            background: #1a3e72;
            border-radius: 10px;
        }
        .modal-story::-webkit-scrollbar-thumb:hover {
            background: #0d2b57;
        }
        
        /* Style untuk close button di modal */
        .close-button {
            background-color: #dc3545 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-size: 1.0em !important;
            margin-top: 20px !important;
            width: 100% !important;
        }
        .close-button:hover {
            background-color: #c82333 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_lda_model():
    """Load LDA model"""
    try:
        model_data = joblib.load('model/lda.joblib')
        
        # Pastikan struktur model lengkap
        if not all(key in model_data for key in ['lda_model', 'dictionary', 'preprocessed_test', 'data_test', 'file_paths_test']):
            st.error("Struktur model LDA tidak lengkap. Pastikan file model berisi semua komponen yang diperlukan.")
            return None
            
        # Tambahkan stop_words dan stemmer jika belum ada
        if 'stop_words' not in model_data:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                from nltk.corpus import stopwords
                import nltk
                
                # Download NLTK data jika belum ada
                try:
                    stopwords.words('indonesian')
                except LookupError:
                    nltk.download('stopwords')
                    nltk.download('punkt')
                
                # Load stopwords
                stop_words_nltk = set(stopwords.words('indonesian'))
                additional_stopwords = set()
                stopword_csv_path = "stopwordbahasa.csv"
                
                if os.path.exists(stopword_csv_path):
                    with open(stopword_csv_path, 'r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            additional_stopwords.update(row)
                
                model_data['stop_words'] = stop_words_nltk.union(additional_stopwords)
                
                # Load stemmer
                factory = StemmerFactory()
                model_data['stemmer'] = factory.create_stemmer()
                
            except Exception as e:
                st.error(f"Gagal memuat komponen preprocessing: {str(e)}")
                return None
                
        return model_data
        
    except FileNotFoundError:
        st.error("File model LDA tidak ditemukan. Pastikan file 'lda.joblib' ada di folder 'model_lda/'")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model LDA: {str(e)}")
        return None

@st.cache_resource
def load_lsi_model():
    """Load LSI model"""
    try:
        model_data = joblib.load('model/lsi_model.joblib')
        
        # Pastikan struktur model lengkap
        required_keys = ['vectorizer', 'svd', 'documents', 'lsi_matrix', 'file_names', 'titles']
        if not all(key in model_data for key in required_keys):
            st.error("Struktur model LSI tidak lengkap. Pastikan file model berisi semua komponen yang diperlukan.")
            return None
            
        return model_data
        
    except FileNotFoundError:
        st.error("File model LSI tidak ditemukan. Pastikan file 'lsi_model.joblib' ada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model LSI: {str(e)}")
        return None

def preprocess_text(text, model_data):
    """Preprocess text for LDA model"""
    if model_data is None:
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in model_data['stop_words'] and len(w) > 2]
    stemmed_tokens = [model_data['stemmer'].stem(w) for w in filtered_tokens]
    return stemmed_tokens

def preprocess_user_input(text, model_data):
    """Preprocess user input for LDA model"""
    tokens = preprocess_text(text, model_data)
    if not tokens:
        simplified_tokens = [w.lower() for w in word_tokenize(text) if len(w) > 1 and w.isalpha()]
        tokens = simplified_tokens
    weighted_tokens = []
    for token in tokens:
        weight = max(1, 5 - len(tokens)) if len(tokens) < 5 else 1
        weighted_tokens.extend([token] * weight)
    return weighted_tokens

def preprocess_lsi_query(text):
    """Preprocess text for LSI model (basic preprocessing)"""
    # Basic preprocessing for LSI
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    # Filter tokens with length > 2
    filtered_tokens = [w for w in tokens if len(w) > 2]
    return ' '.join(filtered_tokens)

def extract_search_keywords(user_input):
    """Ekstrak kata kunci untuk highlighting dari input user"""
    # Tokenize dan bersihkan input
    words = word_tokenize(user_input.lower())
    # Filter kata-kata yang panjangnya > 2 dan hanya huruf
    keywords = [word for word in words if len(word) > 2 and word.isalpha()]
    return keywords

def highlight_keywords_in_text(text, keywords):
    """Highlight kata kunci dalam teks dengan HTML"""
    if not keywords:
        return text
    
    highlighted_text = text
    
    # Sort keywords by length (descending) to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(keyword) + r'\b'
        replacement = f'<span class="highlight-keyword">{keyword}</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text

def create_preview(content, max_length=300, keywords=None):
    cleaned_content = re.sub(r'\s+', ' ', content.strip())
    
    if len(cleaned_content) <= max_length:
        preview = cleaned_content
    else:
        preview = cleaned_content[:max_length]
        last_space = preview.rfind(' ')
        
        if last_space > max_length * 0.8:
            preview = preview[:last_space]
        
        preview = preview + "..."
    
    # Highlight keywords jika ada
    if keywords:
        preview = highlight_keywords_in_text(preview, keywords)
    
    return preview

def format_full_story(content, keywords=None):
    """Format cerita lengkap untuk tampilan yang lebih baik"""
    # Bersihkan spasi berlebih tapi pertahankan semua konten
    cleaned_content = re.sub(r'[ \t]+', ' ', content.strip())
    # Pertahankan semua paragraf
    paragraphs = [p.strip() for p in cleaned_content.split('\n') if p.strip()]
    formatted_content = '\n\n'.join(paragraphs)
    
    # Highlight keywords jika ada
    if keywords:
        formatted_content = highlight_keywords_in_text(formatted_content, keywords)
    
    return formatted_content

def cosine_similarity_manual(vec1, vec2):
    """Manual cosine similarity calculation"""
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def keyword_similarity(query_tokens, doc_tokens):
    """Calculate keyword similarity"""
    if not query_tokens or not doc_tokens:
        return 0.0
    matches = sum(1 for token in query_tokens if token in doc_tokens)
    return matches / len(query_tokens)

def recommend_with_lda(user_input, model_data):
    """Recommend documents using LDA model"""
    if model_data is None:
        return []
    
    user_tokens = preprocess_user_input(user_input, model_data)
    
    if not user_tokens:
        return []
    
    # Langsung menggunakan bag-of-words tanpa TF-IDF
    bow = model_data['dictionary'].doc2bow(user_tokens)
    if len(bow) == 0:
        return []
    
    # Langsung menggunakan LDA dengan BOW
    topic_dist = model_data['lda_model'][bow]
    topic_dist_vec = sparse2full(topic_dist, model_data['lda_model'].num_topics)
    
    results = []
    min_similarity = 0.05
    
    for idx, doc_tokens in enumerate(model_data['preprocessed_test']):
        doc_bow = model_data['dictionary'].doc2bow(doc_tokens)
        if len(doc_bow) > 0:
            # Langsung menggunakan LDA dengan BOW
            doc_topics = model_data['lda_model'][doc_bow]
            doc_topics_vec = sparse2full(doc_topics, model_data['lda_model'].num_topics)
            topic_sim = cosine_similarity_manual(topic_dist_vec, doc_topics_vec)
        else:
            topic_sim = 0.0
        
        keyword_sim = keyword_similarity(user_tokens, doc_tokens)
        combined_score = 0.65 * topic_sim + 0.35 * keyword_sim
        
        if combined_score >= min_similarity:
            content = model_data['data_test'][idx]
            full_path = model_data['file_paths_test'][idx]
            
            # SOLUSI PASTI: Hapus semua path folder dan ekstensi
            # 1. Ambil nama file terakhir (handle both / and \ separators)
            file_name = os.path.basename(full_path.replace("\\", "/"))
            # 2. Hapus ekstensi file
            file_name = os.path.splitext(file_name)[0]
            # 3. Bersihkan dari karakter khusus dan format judul
            title = re.sub(r'[^a-zA-Z0-9\s]', ' ', file_name)  # Hapus karakter non-alphanumeric
            title = ' '.join(title.split())  # Hapus spasi berlebih
            title = title.title()  # Format judul
            
            results.append({
                'title': title,  # Judul sudah bersih
                'content': content,
                'file_name': file_name,
                'score': combined_score,
                'topic_sim': topic_sim,
                'keyword_sim': keyword_sim,
                'index': idx
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    
def recommend_with_lsi(user_input, model_data):
    """Recommend documents using LSI model"""
    if model_data is None:
        return []
    
    # Preprocess query
    processed_query = preprocess_lsi_query(user_input)
    
    if not processed_query.strip():
        return []
    
    try:
        # Transform query using the same vectorizer
        query_tfidf = model_data['vectorizer'].transform([processed_query])
        
        # Transform to LSI space
        query_lsi = model_data['svd'].transform(query_tfidf)
        
        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_lsi, model_data['lsi_matrix']).flatten()
        
        # Get top similar documents
        top_indices = similarities.argsort()[::-1][:10]  # Get top 10 first
        
        results = []
        min_similarity = 0.05  # Minimum similarity threshold
        
        for idx in top_indices:
            similarity_score = similarities[idx]
            
            if similarity_score >= min_similarity:
                # Get document info
                content = model_data['documents'][idx]
                file_name = model_data['file_names'][idx]
                title = model_data.get('titles', [file_name.replace('_', ' ').title()])[idx] if idx < len(model_data.get('titles', [])) else file_name.replace('_', ' ').title()
                
                results.append({
                    'title': title,
                    'content': content,
                    'file_name': file_name,
                    'score': similarity_score,
                    'index': idx
                })
        
        return results[:5]  # Return top 5
        
    except Exception as e:
        st.error(f"Error dalam pencarian LSI: {str(e)}")
        return []

def show_full_story_modal(title, content, file_name, keywords=None):
    """Menampilkan cerita lengkap dalam modal"""
    formatted_story = format_full_story(content, keywords)
    
    st.markdown("---")
    st.markdown(
        f"""
        <div class="modal-content">
            <div class="modal-title">
                📖 {title}
            </div>
            <div style="text-align: center; margin-bottom: 15px; color: #666;">
                📁 File: {file_name}
            </div>
          <div class="modal-story" style="max-height: 300px; overflow-y: scroll;">
                {formatted_story}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_results(results, method_name, search_keywords=None):
    """Display search results"""
    if results:
        st.markdown(f'<p style="color:black; font-size:18px; font-weight:bold; margin-top: -10px;">🎯 Hasil pencarian {method_name} ({len(results)} dongeng ditemukan):</p>', 
                   unsafe_allow_html=True)
        
        # Buat container untuk setiap hasil
        for i, result in enumerate(results, 1):
            title = result['title']
            content = result['content']
            file_name = result['file_name']
            
            # Highlight keywords hanya untuk LDA
            if method_name == "LDA" and search_keywords:
                highlighted_title = highlight_keywords_in_text(title, search_keywords)
                preview = create_preview(content, 300, search_keywords)
            else:
                highlighted_title = title
                preview = create_preview(content, 300)
            
            # Container untuk setiap hasil
            with st.container():
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-title">
                            📚 {i}. {highlighted_title}
                        </div>
                        <div class="result-preview">
                            {preview}
                        </div>
                        <div class="result-info">
                            📁 File: {file_name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Tombol dengan key yang unik berdasarkan index asli dan method
                button_key = f"read_story_{result['index']}_{method_name}_{i}"
                
                if st.button(f"📖 Baca Cerita Lengkap: {title}",
                            key=button_key,
                            type="secondary",
                            use_container_width=True,
                            help="Klik untuk membaca cerita lengkap"):
                    # Set session state untuk menampilkan cerita
                    st.session_state['show_story'] = True
                    st.session_state['selected_story'] = {
                        'title': title,
                        'content': content,
                        'file_name': file_name
                    }
                    # Hanya set search_keywords untuk LDA
                    if method_name == "LDA":
                        st.session_state['search_keywords'] = search_keywords
                    else:
                        st.session_state['search_keywords'] = []
                    st.rerun()
                    
    else:
        st.warning(f"😔 Tidak ditemukan dongeng yang cocok dengan metode {method_name}. Coba kata kunci lain seperti Putri Kerajaan, Cinta dan Kasih Sayang, Keajaiban dan Sihir.")

def main():
    # Initialize session state terlebih dahulu
    if 'show_story' not in st.session_state:
        st.session_state['show_story'] = False
    if 'selected_story' not in st.session_state:
        st.session_state['selected_story'] = None
    if 'search_results_lda' not in st.session_state:
        st.session_state['search_results_lda'] = []
    if 'search_results_lsi' not in st.session_state:
        st.session_state['search_results_lsi'] = []
    if 'current_keywords' not in st.session_state:
        st.session_state['current_keywords'] = []
    if 'search_keywords' not in st.session_state:
        st.session_state['search_keywords'] = []
    
    add_styles()
    
    st.markdown(
        """
        <div class="title-box">
            📚 Dongeng Nusantara Panji Kediri 📚
            <p class="subtitle">Mencari dan menikmati dongeng dari seluruh penjuru negeri</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Tampilkan cerita lengkap jika diminta
    if st.session_state.get('show_story', False) and st.session_state.get('selected_story'):
        story_data = st.session_state['selected_story']
        search_keywords = st.session_state.get('search_keywords', [])
        show_full_story_modal(story_data['title'], story_data['content'], story_data['file_name'], search_keywords)
        
        # Tombol tutup cerita
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("❌ Tutup Cerita", 
                       key="close_story_main",
                       type="secondary", 
                       use_container_width=True):
                st.session_state['show_story'] = False
                st.session_state['selected_story'] = None
                st.rerun()
      
        return  # Keluar dari fungsi jika sedang menampilkan cerita
    
    # Load models
    lda_model_data = load_lda_model()
    lsi_model_data = load_lsi_model()
    
    # Check if at least one model is available
    if lda_model_data is not None or lsi_model_data is not None:
        # Interface pencarian
        with st.container():
            st.markdown('<p class="search-label">Masukkan topik dongeng atau kata kunci pencarian:</p>', unsafe_allow_html=True)
            query = st.text_input(
                "", 
                placeholder="Contoh: Putri Kerajaan",
                label_visibility="collapsed",
                key="search_input"
            )
        
        # Tombol pencarian dengan dua metode
        col1, col2 = st.columns(2)
        with col1:
            lsi_clicked = st.button("🔍 Cari dengan LSI", 
                                  key="lsi_button", 
                                  type="secondary", 
                                  use_container_width=True,
                                  disabled=(lsi_model_data is None))
        with col2:
            lda_clicked = st.button("🔍 Cari dengan LDA", 
                                  key="lda_button", 
                                  type="primary", 
                                  use_container_width=True,
                                  disabled=(lda_model_data is None))
        
        # Handle LSI search
        if lsi_clicked:
            if query.strip():
                if lsi_model_data is not None:
                    with st.spinner("🔍 Mencari dongeng dengan LSI..."):
                        lsi_results = recommend_with_lsi(query, lsi_model_data)
                        st.session_state['search_results_lsi'] = lsi_results
                        st.session_state['search_results_lda'] = []  # Clear LDA results
                        st.session_state['current_keywords'] = []  # No keywords for LSI
                else:
                    st.error("❌ Model LSI tidak tersedia.")
            else:
                st.warning("⚠️ Mohon masukkan kata kunci pencarian terlebih dahulu.")
        
        # Handle LDA search
        if lda_clicked:
            if query.strip():
                if lda_model_data is not None:
                    with st.spinner("🔍 Mencari dongeng dengan LDA..."):
                        # Extract keywords untuk highlighting (hanya untuk LDA)
                        search_keywords = extract_search_keywords(query)
                        
                        lda_results = recommend_with_lda(query, lda_model_data)
                        st.session_state['search_results_lda'] = lda_results
                        st.session_state['search_results_lsi'] = []  # Clear LSI results
                        st.session_state['current_keywords'] = search_keywords
                else:
                    st.error("❌ Model LDA tidak tersedia.")
            else:
                st.warning("⚠️ Mohon masukkan kata kunci pencarian terlebih dahulu.")
        
        # Tampilkan hasil pencarian
        if st.session_state.get('search_results_lda', []):
            display_results(
                st.session_state['search_results_lda'], 
                "LDA", 
                st.session_state.get('current_keywords', [])
            )
        elif st.session_state.get('search_results_lsi', []):
            display_results(
                st.session_state['search_results_lsi'], 
                "LSI", 
                []  # No keywords for LSI
            )
                
    else:
        st.error("❌ Tidak ada model yang tersedia. Pastikan file model ada di direktori yang benar:")
        st.info("💡 Jika Anda menjalankan ini secara lokal, pastikan Anda telah menjalankan script pelatihan model terlebih dahulu")

if __name__ == "__main__":
    main()
