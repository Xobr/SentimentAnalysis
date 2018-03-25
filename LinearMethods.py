from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn import metrics

import pandas as pn
import numpy as np


def getModel():
    return Pipeline([
        ('vectorizer', CountVectorizer()),
        ('logisticRegression', LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced',max_iter=1000))
    ])

def getModelNB():
    return Pipeline([
        ('vectorizer', CountVectorizer()),
        ('NB', MultinomialNB())
    ])

df = pn.DataFrame.from_csv('train.tsv', sep='\t')
trainset, testset = train_test_split(df, test_size=0.15)
y = np.asarray(trainset['Sentiment'])
model = getModel()
model.fit(trainset['Phrase'], y)
pred = model.predict(testset['Phrase'])

print(metrics.classification_report(np.asarray(testset['Sentiment']), pred))
print("\n")
print('Accuracy', metrics.accuracy_score(np.asarray(testset['Sentiment']), pred))

# print(model)


