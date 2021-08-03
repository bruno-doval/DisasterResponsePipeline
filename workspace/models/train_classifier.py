import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords'])
import joblib
 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

"""
To run ML pipeline that trains classifier and saves

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
"""


def load_data(database_filepath):
    """
    INPUT:
    database_filepath: the file path to the database

    OUTPUT:
    X: features of the message df
    y: classifications of the messages
    col_list: list of classification columns
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))


    df = pd.read_sql("SELECT * FROM Message", engine)

    X = df['message']
    y = df.drop(columns=['id','message','original','genre'])
    col_list = list(y.columns)
    return X,y,col_list


def tokenize(text):
    """
    INPUT:
    text: text to be tokenized

    OUTPUT:
    clean_tokens: text tokenized
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    



def build_model():

    """
    builds a pipeline to be used for modeling a df
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=10)))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),

        'clf__estimator__min_samples_split': [2, 3, 4]

    }


    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model: model to be applied on the df
    X_test: part of the df to be used as featues
    Y_text: response part of the df
    category_names: names pf the categories

    OUTPUT:
    print the repport comparing the test response data (Y) with the model applied to the test data (X)
    """

    y_pred = model.best_estimator_.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred,columns=Y_test.columns)

    for col in category_names:
        print(classification_report(Y_test[col], y_pred_df[col]))


def save_model(model, model_filepath):
    """
    Saves the model into a pickle file

    INPUT:
    model: model to be saved
    model_filepath: file path to save the model on
    """
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()