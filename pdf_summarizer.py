import streamlit as st
import PyPDF2
import re
from collections import Counter
import heapq
import datetime

# Set page config
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

# Try to import and setup NLTK, fallback to simple methods if it fails
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    @st.cache_resource
    def setup_nltk():
        """Setup NLTK data with fallback options"""
        try:
            # Try the new punkt_tab first
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except:
                nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to old punkt
            try:
                nltk.data.find('tokenizers/punkt')
            except:
                nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except:
            nltk.download('stopwords', quiet=True)
    
    # Setup NLTK
    setup_nltk()
    NLTK_AVAILABLE = True
    
except Exception as e:
    st.warning(f"NLTK setup failed: {e}. Using simple tokenization.")
    NLTK_AVAILABLE = False

# Simple fallback tokenizers (if NLTK fails)
def simple_sent_tokenize(text):
    """Simple sentence tokenization without NLTK"""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences

def simple_word_tokenize(text):
    """Simple word tokenization without NLTK"""
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    return words

# Basic stopwords list (fallback if NLTK fails)
BASIC_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
    'should', 'now', 'could', 'would', 'may', 'might', 'must', 'shall'
}

# Smart tokenization functions that use NLTK if available, fallback otherwise
def tokenize_sentences(text):
    """Tokenize sentences using NLTK if available, simple method otherwise"""
    if NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except:
            return simple_sent_tokenize(text)
    else:
        return simple_sent_tokenize(text)

def tokenize_words(text):
    """Tokenize words using NLTK if available, simple method otherwise"""
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except:
            return simple_word_tokenize(text)
    else:
        return simple_word_tokenize(text)

def get_stopwords():
    """Get stopwords using NLTK if available, basic set otherwise"""
    if NLTK_AVAILABLE:
        try:
            return set(stopwords.words('english'))
        except:
            return BASIC_STOPWORDS
    else:
        return BASIC_STOPWORDS

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
    return text.strip()

def extractive_summarize(text, num_sentences=5):
    """Create extractive summary using frequency-based scoring"""
    try:
        # Clean text
        text = clean_text(text)
        
        # Tokenize into sentences
        sentences = tokenize_sentences(text)
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Get stopwords
        stop_words = get_stopwords()
        
        # Calculate word frequency (excluding stopwords)
        word_freq = {}
        for sentence in sentences:
            words = tokenize_words(sentence)
            for word in words:
                if word not in stop_words and len(word) > 2 and word.isalnum():
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = tokenize_words(sentence)
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            # Average score per word, with bonus for sentence position
            if word_count > 0:
                avg_score = score / word_count
                # Give slight bonus to sentences at beginning and end
                position_bonus = 1.0
                if i < len(sentences) * 0.3:  # First 30%
                    position_bonus = 1.2
                elif i > len(sentences) * 0.7:  # Last 30%
                    position_bonus = 1.1
                
                sentence_scores[sentence] = avg_score * position_bonus
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        # Arrange in original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return " ".join(summary_sentences)
    
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def simple_abstractive_summarize(text, target_length=200):
    """Simple abstractive summarization using keyword-based sentence selection"""
    try:
        text = clean_text(text)
        sentences = tokenize_sentences(text)
        
        current_length = len(text.split())
        if current_length <= target_length:
            return text
        
        # Get stopwords
        stop_words = get_stopwords()
        
        # Extract important keywords
        words = tokenize_words(text)
        
        # Filter and count words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2 and word.isalnum()]
        word_freq = Counter(filtered_words)
        
        # Get most important words
        important_words = set([word for word, _ in word_freq.most_common(30)])
        
        # Score sentences based on important words
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(tokenize_words(sentence))
            
            # Calculate importance score
            importance_score = len(sentence_words.intersection(important_words))
            sentence_length = len(sentence.split())
            
            # Avoid very short or very long sentences
            if 5 <= sentence_length <= 50 and importance_score > 0:
                # Normalize by sentence length
                normalized_score = importance_score / sentence_length
                scored_sentences.append((normalized_score, sentence, sentence_length))
        
        # Sort by score (highest first)
        scored_sentences.sort(reverse=True)
        
        # Build summary within target length
        summary_parts = []
        total_words = 0
        
        for score, sentence, length in scored_sentences:
            if total_words + length <= target_length:
                summary_parts.append(sentence)
                total_words += length
            elif total_words < target_length * 0.8:  # If we haven't reached 80% of target
                # Try to fit a shorter version of the sentence
                words_remaining = target_length - total_words
                if words_remaining >= 10:
                    sentence_words = sentence.split()
                    truncated = " ".join(sentence_words[:words_remaining-1]) + "..."
                    summary_parts.append(truncated)
                    break
            else:
                break
        
        return " ".join(summary_parts)
    
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def main():
    st.title("📄 PDF Text Summarizer")
    st.markdown("Upload a PDF file to get an intelligent summary using natural language processing")
    
    # Show NLTK status
    if NLTK_AVAILABLE:
        st.success("✅ Advanced NLTK tokenization enabled")
    else:
        st.info("ℹ️ Using simple tokenization (NLTK not available)")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses **Natural Language Processing** to summarize PDF documents.
        
        **Features:**
        - 📤 Drag & drop PDF upload
        - 🔍 Automatic text extraction
        - 📊 Two summarization methods
        - 📈 Summary statistics
        - 💾 Download results
        - 🔧 Smart NLTK integration with fallbacks
        """)
        
        st.header("🔧 Methods")
        st.markdown("""
        **Extractive:** Selects the most important sentences from the original text based on word frequency and position.
        
        **Abstractive:** Creates condensed text by selecting key sentences that contain the most important concepts.
        """)
        
        st.header("🛠️ Technical Info")
        if NLTK_AVAILABLE:
            st.success("NLTK: ✅ Active")
        else:
            st.warning("NLTK: ⚠️ Fallback mode")
    
    # Instructions
    with st.expander("📋 Quick Start Guide", expanded=False):
        st.markdown("""
        1. **📁 Upload PDF**: Click below or drag & drop your file
        2. **⚙️ Choose Settings**: Select summarization method and length
        3. **🚀 Generate**: Click the button to create your summary
        4. **💾 Download**: Save your summary for later use
        
        **Supported formats:** PDF files up to 10MB
        **Languages:** English (optimized)
        **Note:** Works with or without NLTK dependencies!
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose a PDF file",
        type="pdf",
        help="Select a PDF file to summarize (max 10MB)"
    )
    
    if uploaded_file is not None:
        # File information
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Size: {file_size_mb:.1f} MB")
        with col2:
            if file_size_mb > 10:
                st.error("❌ File too large! Please use files under 10MB")
                return
            else:
                st.info("✅ File size OK")
        
        # Extract text
        with st.spinner("🔍 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text and len(text.strip()) > 0:
            st.success("✅ Text extracted successfully!")
            
            # Text statistics
            word_count = len(text.split())
            sentence_count = len(tokenize_sentences(text))
            char_count = len(text)
            
            # Display statistics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Words", f"{word_count:,}")
            with col2:
                st.metric("📄 Sentences", f"{sentence_count:,}")
            with col3:
                st.metric("🔤 Characters", f"{char_count:,}")
            
            # Show text preview
            with st.expander("📖 Text Preview", expanded=False):
                preview_length = 1500
                preview_text = text[:preview_length]
                if len(text) > preview_length:
                    preview_text += f"\n\n... ({len(text) - preview_length:,} more characters)"
                st.text_area("Extracted text:", preview_text, height=300, disabled=True)
            
            # Summary settings
            st.subheader("⚙️ Summary Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox(
                    "📊 Summarization Method",
                    ["Extractive", "Abstractive"],
                    help="Extractive: Selects key sentences. Abstractive: Creates condensed content."
                )
            
            with col2:
                if method == "Extractive":
                    max_sentences = min(20, max(3, sentence_count // 5))
                    param = st.slider(
                        "🎯 Number of sentences", 
                        3, 
                        max_sentences, 
                        min(7, max_sentences),
                        help="How many sentences in the summary"
                    )
                else:
                    max_words = min(800, max(100, word_count // 3))
                    param = st.slider(
                        "🎯 Target word count", 
                        100, 
                        max_words, 
                        min(250, max_words),
                        help="Approximate number of words in summary"
                    )
            
            # Generate summary button
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                if word_count < 100:
                    st.warning("⚠️ Text might be too short for effective summarization.")
                    st.info("💡 Try a document with at least 100 words for better results.")
                
                with st.spinner(f"🤖 Creating {method.lower()} summary..."):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    
                    # Generate summary
                    if method == "Extractive":
                        progress_bar.progress(50)
                        summary = extractive_summarize(text, param)
                    else:
                        progress_bar.progress(50)
                        summary = simple_abstractive_summarize(text, param)
                    
                    progress_bar.progress(100)
                    progress_bar.empty()
                    
                    # Display summary
                    st.subheader("📝 Summary")
                    
                    # Summary in a nice container
                    with st.container():
                        st.markdown("---")
                        st.write(summary)
                        st.markdown("---")
                    
                    # Summary statistics
                    summary_words = len(summary.split())
                    summary_chars = len(summary)
                    compression_ratio = (1 - summary_words/word_count) * 100 if word_count > 0 else 0
                    
                    # Statistics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Summary Words", f"{summary_words:,}")
                    with col2:
                        st.metric("Compression", f"{compression_ratio:.1f}%")
                    with col3:
                        reading_time = max(1, summary_words // 200)  # Average reading speed
                        st.metric("Reading Time", f"{reading_time} min")
                    
                    # Download section
                    st.subheader("💾 Download Summary")
                    
                    # Prepare download content
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    nltk_status = "NLTK Enhanced" if NLTK_AVAILABLE else "Simple Tokenization"
                    
                    download_content = f"""PDF SUMMARY REPORT
{'='*50}

Original File: {uploaded_file.name}
Summarization Method: {method}
Processing Mode: {nltk_status}
Generated on: {timestamp}

DOCUMENT STATISTICS:
- Original length: {word_count:,} words
- Summary length: {summary_words:,} words
- Compression ratio: {compression_ratio:.1f}%
- Reading time: ~{reading_time} minute(s)

SUMMARY:
{'='*50}

{summary}

{'='*50}
Generated by PDF Summarizer App
"""
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Summary Report",
                        data=download_content,
                        file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    # Success message
                    st.success("🎉 Summary generated successfully!")
        
        elif text is not None:
            st.error("❌ The PDF appears to be empty or contains no readable text.")
            st.info("💡 Try a different PDF file with text content.")
        else:
            st.error("❌ Could not extract text from the PDF.")
            st.info("💡 Make sure the file is a valid PDF with readable text.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔧 <strong>PDF Text Summarizer</strong> | Built with ❤️ using Streamlit</p>
        <p><small>✨ Smart NLTK integration • Works offline • Privacy-friendly</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
