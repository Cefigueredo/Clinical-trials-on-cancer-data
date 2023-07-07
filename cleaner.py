# Limpieza de datos
# Instalación de librerias
import re
import unicodedata

import contractions
import inflect
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = ""
data_t = DATA


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_words.append(word.lower())
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r"[^\w\s]", "", word)
        if new_word != "":
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words
    with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    stop = stopwords.words("english")

    for word in words:
        if word not in (stop):
            new_words.append(word)

    return new_words


def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


# Eliminación registros con ausencias
data_t = data_t.dropna()
# Eliminación de registros duplicados.
data_t = data_t.drop_duplicates()
data_t["label"].value_counts()

# Tokenización

data_t["study_and_condition"] = data_t["study_and_condition"].apply(
    contractions.fix
)  # Aplica la corrección de las contracciones

data_t["words"] = (
    data_t["study_and_condition"].apply(word_tokenize).apply(preprocessing)
)  # Aplica la eliminación del ruido

new_words = []
for word in data_t["words"]:
    new_words = word.remove("study")
    new_words = word.remove("interventions")
    data_t["words"] = data_t["words"].replace(new_words)
data_t.head()

# Normalización

lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()
stop = stopwords.words("english")


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def stem_words(words):
    """Stem words in list of tokenized words"""
    # https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    porter = PorterStemmer()
    new_words = []
    for word in words:
        new_words.append(porter.stem(word))
    return new_words


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    wnl = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(wnl.lemmatize(word))
    return new_words


def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas


data_t["words"] = data_t["words"].apply(
    stem_and_lemmatize
)  # Aplica lematización y Eliminación de Prefijos y Sufijos.

# Selección de campos

data_t["words"] = data_t["words"].apply(lambda x: " ".join(map(str, x)))

# TF_IDF

# Source: (https://towardsdatascience.com/natural-language-processing-feature-
# engineering-using-tf-idf-e8b9d00e7e76)
vectorizer = TfidfVectorizer()
allDocs = []
for word in data_t["words"]:
    allDocs.append(word)
vectors = vectorizer.fit_transform(allDocs)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
data_tfidf = pd.DataFrame(denselist, columns=feature_names)
data_tfidf.head()
