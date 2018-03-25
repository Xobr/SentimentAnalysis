import numpy as np
import pandas as pn
import datetime as dt
from sklearn.pipeline import Pipeline
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.layers import Dense, Activation , Input,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
from SubmissionGenerator import submission
from enum import Enum
import sys

class typeModel(Enum):
    Simply = 1
    Lstm = 2

max_words = 18227
num_classes = 5
batch_size = 32
epochs = 1
df = pn.DataFrame.from_csv('train.tsv', sep='\t')
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(df['Phrase'])

modelType = typeModel.Simply

class kerasModel:

    def __init__(self, typeOfModel: typeModel):
        if(typeOfModel == typeModel.Simply):
            self.model = self.create_model()
        elif(typeOfModel == typeModel.Lstm):
            self.model = self.create_lstm_model()
        else:
            self.model = self.create_model()

    def create_model(self):
        print ('Creating model...')
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def create_lstm_model(self):
        print ('Creating model...')

        max_features = 100000
        maxlen = 100

        # model = Sequential()
        # model.add(Embedding(max_features, 128, input_length=maxlen))
        # model.add(LSTM(64, return_sequences=True))
        # model.add(LSTM(64))
        # model.add(Dropout(0.5))
        # model.add(Dense(5))
        # model.add(Activation('sigmoid'))
        #
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               class_mode="binary")

        model = Sequential()
        model.add(Embedding(100, output_dim=256))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def transform(self):
        return self.model

    def fit(self, x, y):
        self.model.fit(x,y,batch_size=batch_size,
                    epochs= epochs )

        return self.model

    def predict(self,X):
        return self.model.predict(X)

class bagOfWords:

    def fit(self, x , y = None):
      return self

    def transform(self,x):
        return tokenizer.texts_to_matrix(x, mode='binary')

    def predict(self):
        return self

class lstmPreparer:
    def fit(self, x , y = None):
      return self

    def transform(self,x):
        raw = tokenizer.texts_to_sequences(x)
        return sequence.pad_sequences(raw, maxlen=100)

    def predict(self):
        return self

def getModel(tp: typeModel = modelType) -> Pipeline:
    if(tp == typeModel.Simply):
        return Pipeline([
        ('tranformation', bagOfWords()),
        ('model', kerasModel(tp))
        ])
    elif(tp == typeModel.Lstm):
        return Pipeline([
            ('tranformation', lstmPreparer()),
            ('model', kerasModel(tp))
        ])

def saveModel(model):
    kerasModel = model.named_steps['model'].model
    name = 'Models/{0}_epoch_lstm_evaluate'.format(epochs)
    kerasModel.save(name)

def evalueteModel():
    train1, test1 = train_test_split(df, test_size=0.001)
    train, test = train_test_split(test1, test_size=0.02)
    x_train = train['Phrase']
    x_test = test['Phrase']
    y_train = to_categorical(train['Sentiment'], num_classes)
    y_test = test['Sentiment']
    model = getModel()
    model.fit(x_train, y_train)
    print('model is fited')
    pred_raw = model.predict(x_test)
    pred = [list(ls).index(max(ls)) for ls in pred_raw]
    acc = metrics.accuracy_score(np.asarray(y_test), pred)
    print(acc)

    with open('NNLogs.txt','a') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print('\n')
        print('---------------------')
        print('Model info')
        print('date: {0}'.format(dt.datetime.now()))
        model.named_steps['model'].model.summary()
        print('accuracy: {0}'.format(acc))
        print('---------------------')
        sys.stdout = orig_stdout
    saveModel(model)
    print('model is saved')

def trainSavaeModel():
    x = df['Phrase']
    y = to_categorical(df['Sentiment'], num_classes)
    model = getModel()
    model.fit(x.tolist(), y)
    kerasModel = model.named_steps['model'].model
    name = '{0}_epoch'.format(epochs)
    kerasModel.save('Models/{0}'.format(name))
    print('model saved')
    submission.doSubmission(max_words,name,model = kerasModel,tokenizer = tokenizer)
    print('submission done')

print(dt.datetime.now())
trainSavaeModel()
print(dt.datetime.now())