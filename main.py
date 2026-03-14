import nltk
# Download both versions to ensure compatibility
try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)

nltk.download('stopwords', quiet=True)
print("NLTK data downloaded successfully!")
