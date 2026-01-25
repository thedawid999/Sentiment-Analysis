import re
import string
import nltk
import joblib
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

nltk.download('averaged_perceptron_tagger_eng')
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "nor", "n't", "never", "hardly", "barely"}
STOPWORDS = STOPWORDS - NEGATIONS

lemmatizer = WordNetLemmatizer()

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)

def remove_emojis(token_list):
    if isinstance(token_list, list):
        return [emoji_pattern.sub("", token) for token in token_list]
    return token_list

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return "n"

def my_tokenizer(text):
    return text

def preprocess(title, review):
    text = title + " " + review
    # 1. lowercase
    text = text.lower()

    # 2. tokenization
    tokens = word_tokenize(text)

    # 3. remove emojis
    tokens = remove_emojis(tokens)

    # 4. remove stopwords + punctuation
    tokens = [
        t for t in tokens
        if t not in STOPWORDS and t not in string.punctuation
    ]

    # 5. lemmatization + POS
    pos_tags = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    tfidf = joblib.load("results/tfidf_vectorizer.joblib")
    data_vec = tfidf.transform(tokens)

    return data_vec


print("Choose model:")
print("1 - Logistic Regression")
print("2 - Naive Bayes")

choice = input("Your choice: ").strip()

if choice == "1":
    model = joblib.load("results/logistic_regression.joblib")
elif choice == "2":
    model = joblib.load("resuluts/multinomial_nb.joblib")
else:
    raise ValueError("Invalid choice")

title = input("Enter review title: ")
review = input("Enter review text: ")

data_preprocessed = preprocess(title, review)
prediction = model.predict(data_preprocessed)[0]

print(f"\nPredicted sentiment: {prediction}")
