# Text processing

import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple was founded by Steve Jobs in 1976."
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_, token.dep_)
    print(token.is_stop, token.is_punct)

for ent in doc.ents:
    print(ent.text, ent.label_)


# using EntityRuler

from spacy.pipeline import EntityRuler

ruler = EntityRuler(nlp)

patterns = [
    {"label": "INVOICE_ID", "pattern": [{"LOWER": "invoice"}, {"IS_DIGIT": True}]},
    {"label": "PHONE_NUMBER", "pattern": [{"REGEX": r"^\d{10}$"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "python"}]},
    {"label": "PRODUCT", "pattern": [{"POS": "ADJ"}, {"POS": "NOUN"}]},
]

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

text = (
    "Invoice 12345 has phone number 1234567890 and skill python and product good book"
)
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)


# Bag of words
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text])
print(vectorizer.get_feature_names_out())
print(X.toarray())

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform([text])
print(tfidf.get_feature_names_out())
print(X.toarray())

# Word2Vec
import gensim
from gensim.models import Word2Vec

sentences = [[word.text for word in doc]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv["apple"])

# Binary Text classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


texts = ["I love this movie", "This film is terrible"]
labels = [1, 0]  # 1 = positive, 0 = negative

X = TfidfVectorizer().fit_transform(texts)

model = LogisticRegression().fit(X, labels)
print(model.predict(X))

# Multi Class Text classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


texts = [
    "The team won the match",
    "The election results are out",
    "New AI technology is released",
    "Government passed a new law",
    "Player scored a goal",
]

# 0 = Sports, 1 = Politics, 2 = Technology
labels = [0, 1, 2, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)


model = LogisticRegression(multi_class="multinomial", max_iter=1000)
model.fit(X, labels)

test = ["AI is changing the world"]
test_vector = vectorizer.transform(test)

model.predict(test_vector)


# Multi Label Text classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


texts = [
    "AI in healthcare",
    "Football match analysis",
    "New government AI policy",
    "Medical sports training",
]


labels = [
    ["Technology", "Health"],
    ["Sports"],
    ["Technology", "Politics"],
    ["Health", "Sports"],
]

mlb = MultiLabelBinarizer()  # Convert labels to binary matrix
Y = mlb.fit_transform(labels)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, Y)

test = ["AI in sports medicine"]
test_vector = vectorizer.transform(test)

prediction = model.predict(test_vector)

print("Predicted labels:", mlb.inverse_transform(prediction))


# for day 9 and 10 refer corresponding day's codes