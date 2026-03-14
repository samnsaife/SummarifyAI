# 📄 SummarifyAI -- Smart PDF Summarizer

An intelligent **PDF text summarization web app** built with
**Streamlit** and **Natural Language Processing (NLP)**.\
It extracts text from PDF files and generates both **extractive** and
**abstractive summaries**, with smart fallbacks for environments where
NLTK is not available.

------------------------------------------------------------------------

## 🚀 Features

-   📤 **Upload PDFs** (up to 10MB)\
-   🔍 **Automatic text extraction** using PyPDF2\
-   📊 **Two summarization modes**:
    -   **Extractive**: Selects key sentences based on frequency and
        position\
    -   **Abstractive**: Condenses text using important concepts\
-   ⚡ **Fallback system**: Works with or without NLTK installed\
-   📈 **Text statistics**: word count, sentence count, compression
    ratio, and reading time\
-   💾 **Downloadable summary reports**\
-   🎨 **Modern Streamlit interface** with progress bars and metrics

------------------------------------------------------------------------

## 🏗️ Architecture

                    STREAMLIT WEB INTERFACE
       File Upload → Text Extraction → Processing → Summarization
                                 │
                                 ▼
                    CORE PROCESSING LAYER
         PDF Processing | Text Analysis | Summarization Engine
         • PyPDF2       | • Tokenization| • Extractive Method
         • Cleaning     | • Word Freq.  | • Abstractive Method
                        | • Scoring     | • Sentence Selection
                                 │
                                 ▼
                    NLP PROCESSING LAYER
           NLTK Enhanced      |     Fallback System
           • Tokenization     |  • Regex Tokenization
           • Stopwords        |  • Basic Stopwords
                              |  • Error Handling

------------------------------------------------------------------------

## 📦 Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/your-username/summarifyai.git
    cd summarifyai
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  (Optional) Download NLTK resources:

    ``` bash
    python main.py
    ```

------------------------------------------------------------------------

## ▶️ Usage

Run the Streamlit app:

``` bash
streamlit run pdf_summarizer.py
```

Then open the link shown in the terminal (usually
`http://localhost:8501/`).

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── pdf_summarizer.py   # Main Streamlit application
    ├── main.py             # Helper script to download NLTK data
    ├── architecture.txt    # Architecture diagram (ASCII format)
    ├── requirements.txt    # Dependencies
    └── README.md           # Project documentation

------------------------------------------------------------------------

## ⚙️ Requirements

-   Python 3.8+
-   Streamlit 1.28.2\
-   PyPDF2 3.0.1\
-   NLTK 3.8.1

------------------------------------------------------------------------

## 📝 Example Workflow

1.  Upload a PDF file (research paper, report, etc.)\
2.  Choose summarization method (**Extractive** or **Abstractive**)\
3.  Adjust settings (number of sentences or word limit)\
4.  Click **Generate Summary**\
5.  Download your summary as a `.txt` report

------------------------------------------------------------------------

## 💡 Future Enhancements

-   Multi-language support\
-   Integration with Hugging Face Transformers for advanced abstractive
    summaries\
-   Support for DOCX and TXT files
