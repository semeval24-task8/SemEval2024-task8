from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVR, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
import pandas
from statistics import mean
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import numpy as np
import spacy
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

np.random.seed(101)

tasks = {}
tasks["A_mono"] = {}
tasks["A_mono"]["train_data"] = "/Users/phenningsson/tfidfcalc/SubtaskA/subtaskA_train_monolingual.jsonl"
tasks["A_mono"]["dev_data"] = "/Users/phenningsson/tfidfcalc/SubtaskA/subtaskA_dev_monolingual.jsonl"

train_data = pandas.read_json(tasks["A_mono"]["train_data"], lines=True)
dev_data = pandas.read_json(tasks["A_mono"]["dev_data"], lines=True)
    
#fraction = 0.05

#train_data = train_data.sample(frac=fraction)
#dev_data = dev_data.sample(frac=fraction)

X_train = train_data["text"]
y_train = train_data["label"]
X_dev = dev_data["text"]
y_dev = dev_data["label"]

X_train_list = X_train.to_numpy()

#naive bayes classifier
clf = MultinomialNB()
  
tfidf_vectorizer = TfidfVectorizer()  
tfidf_matr = tfidf_vectorizer.fit_transform(X_train_list)

#count = CountVectorizer(ngram_range=(1,1))
#tf_idf = TfidfVectorizer(ngram_range=(1,1))
#char = CountVectorizer(analyzer='char_wb', ngram_range=(1,1), min_df=3)
    
    
pipe = Pipeline([('cls', clf)])

# confusion_matrx 
#y_pred = cross_val_predict(pipe, tfidf_matr, y_train)
#conf_matrix = confusion_matrix(y_train, y_pred)
#fig, ax = plt.subplots(figsize=(11,11))
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.title('Confusion Matrix.')
#plt.show()


    # ADD X_TRAIN_MATR HERE MATTHIJS
y_pred = cross_val_predict(pipe, tfidf_matr, y_train)
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

#feature_names = tfidf_vectorizer.get_feature_names_out()
#feature_names = tfidf_vectorizer.vocabulary_
#print("TF-IDF Features:", feature_names)
#np.savetxt(tfidf_matr, header=",".join(tfidf_vectorizer.get_feature_names_out()))

# Convert the feature matrix to a DataFrame
#tfidf_df = pandas.DataFrame(tfidf_matr.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save the DataFrame as a CSV file
#tfidf_df.to_csv('tfidf_features.csv', index=False)

#sp.save_npz('tfidf_features_A_mono_train.npz', tfidf_matr)
