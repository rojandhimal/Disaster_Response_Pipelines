# import libraries
import nltk
import numpy as np
import sqlite3
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import sys


def load_data(database_filepath):
    """
    This function load data from database from given database_filepath.
    process and slit the data as label and features (x ,y, categoory_name)
    
    Input args:
    Database file path
    
    Output :
    returnd seperated feature and labels (X,y,categoory_name)
    
    """
    table_name  = "Disaster"
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)
    
    # drop columns with null
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X,y,category_names


def tokenize(text):
    """
    This function take text and return token of words
    
    Input args:
    Text 
    
    Output:
    List of clean token
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
    Build Multiclass Randomforest model
    Create pipeline
    Hypertune model 
    
    Input :
    N/A
    
    Output:
    returns Hypertuned model
    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}


    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluate the model and return the classification and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    Y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))
    


def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.as_matrix(), Y_train.as_matrix())
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        

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