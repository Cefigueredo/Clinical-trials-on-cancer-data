import nltk

# Punkt permite separar un texto en frases.
nltk.download("punkt")

# Descarga todas las palabras vacias, es decir, aquellas que no aportan nada al significado del texto
# ¿Cuales son esas palabras vacías?

nltk.download("stopwords")

# Descarga de paquete WordNetLemmatizer, este es usado para encontrar el lema de cada palabra
# ¿Qué es el lema de una palabra? ¿Qué tan dificil puede ser obtenerlo, piensa en el caso en que tuvieras que escribir la función que realiza esta tarea?
nltk.download("wordnet")

# Instalación de librerias
import pandas as pd
import numpy as np
import sys
import seaborn as sns
from pandas_profiling import ProfileReport

import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from pandas.core.dtypes.generic import ABCIndex
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    plot_precision_recall_curve,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import validation_curve

# Para búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV

# Para la validación cruzada
from sklearn.model_selection import KFold

# Para usar KNN como clasificador
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Se cargan los datos.
data = pd.read_csv(
    "clinical_trials_on_cancer_data_clasificacion.csv",
    sep=",",
    encoding="utf-8",
    index_col=None,
    low_memory=False,
)

# Limpieza de datos

# Es recomendable que todos los pasos preparación se realicen sobre otro archivo.
data_t = data


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
    """Replace all interger occurrences in list of tokenized words with textual representation"""
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
    lancaster = LancasterStemmer()
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

# Source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
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

# Modelado SVM (Support Vector Machine)

# Se selecciona la variable objetivo, en este caso "label".
Y = data_t["label"]
# Se pasan como inputs los valores a los que se les aplicó TF_IDF
X = data_tfidf

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"],
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, Y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(Y_test, grid_predictions))
