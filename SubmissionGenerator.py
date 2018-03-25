import keras
from keras.preprocessing.text import Tokenizer
import pandas as pn
from keras.models import load_model

class submission:

    @staticmethod
    def doSubmission(max_words, name , tokenizer = None, model = None):

        if model is None:
            model = load_model(name)
        df = pn.read_csv('train.tsv',sep='\t')
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(df['Phrase'])

        # x_train = tokenizer.texts_to_matrix(train['Phrase'], mode='binary')
        data = pn.read_csv('test.tsv',sep= '\t')
        x_data = tokenizer.texts_to_matrix(data['Phrase'], mode='binary')
        pred_raw = model.predict(x_data)
        pred = [list(ls).index(max(ls)) for ls in pred_raw]

        res = pn.DataFrame(data = {'PhraseId': data['PhraseId'],'Sentiment': pred})
        subName = 'Submisstions/{0}_submisstion.csv'.format(name)
        res.to_csv(subName,index=False)

# submission.doSubmission(18227,'4_epoch')
# print('Done')