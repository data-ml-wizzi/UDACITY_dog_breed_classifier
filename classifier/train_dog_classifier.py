import sys
import pandas as pd
import numpy as np

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    """
    Load stored Messages and Categories from SQL database
    
    Parameter:
        database_filepath: Path to the database file
    
    Return:
        X: Input data
        y: Output data
    """
    
    # load data from database
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql('SELECT * FROM TAB_DR', engine)


    # Define X (Input) and y (Output)
    X = df.message
    y = df.drop(columns=['id', 'message', 'original', 'genre']) 
    
    return X, y

def tokenize(text):
    """
    Function to clean text massages. Data cleaning is carried out in four steps:
        1. Delete URLs
        2. Normalize
        3. Tokenize text in words
        4. Remove stop words
        5. Lemmatize tokenized words and Normalize
    
    Parameter:
        text: The text string to be cleaned.
    
    Return:
        clean_tokens: Cleaned token
    """
    
    # Step 1: Delete URLs:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Step 2: Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Step 3: Tokenize text in words:
    tokens = word_tokenize(text)

    # Step 4: Remove stop words
    words = [t for t in tokens if t not in stopwords.words("english")]
    
    # Step 5: Lemmatize tokenized words and Normalize:
    clean_tokens = []
    for word in words:
        clean_tok = WordNetLemmatizer().lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)
      
    return clean_tokens
    
def build_model():
    """
    Function to build the Machine Learning Model for prediction. AdaBoost is used.
    
    Parameter:
        NONE
    
    Return:
        model: As pipeline for training and prediction
    """    

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.2,
                                                         n_estimators=100),
                                                         n_jobs=10))
    ])
    
    
    return model

def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate the Machine Learning Model.
    
    Parameter:
        model: Builded Model
        X_test: Input test data
        Y_test: Output test data
    
    Return:
        NONE
    """ 
    
    # Predict values with model:
    y_pred = model.predict(X_test)

    # classification report on test data
    print(classification_report(Y_test.values,
                                y_pred,
                                target_names=Y_test.columns.values))
    
    return

def save_model(model, model_filepath):
    """
    Save the model into a PICKLE-File
    
    Parameter:
        model: Model to save
        model_filepath: Path 
    
    Return:
        NONE
    """     
    pickle.dump(model, open(model_filepath, 'wb') )
   
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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