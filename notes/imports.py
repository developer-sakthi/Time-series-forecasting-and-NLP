# =========================
# NLP & Text Processing
# =========================
import spacy
from spacy.pipeline import EntityRuler

# =========================
# Vectorization
# =========================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Classification Models
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# =========================
# Word Embeddings
# =========================
import gensim
from gensim.models import Word2Vec

# =========================
# Time Series – Stationarity Tests
# =========================
from statsmodels.tsa.stattools import adfuller, kpss

# =========================
# ARIMA / SARIMAX Models
# =========================
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================
# ACF / PACF Plots
# =========================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =========================
# Data Handling
# =========================
import pandas as pd
