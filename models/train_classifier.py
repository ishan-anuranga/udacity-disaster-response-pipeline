"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl
Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

import sys
import os
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import pickle
import re
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from starting_verb_extractor import StartingVerbExtractor

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = database_filepath.replace(".db","").split('/')[1] + "_table"
    df = pd.read_sql_table(table_name, engine)
    
    # Dropping child_alone since it doesn't contain only zeros 
    del df['child_alone']
    
    # Since related variable should be 1 or 0, Let's convert value 2 to 1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # Seperate features and target
    X = df['message']
    y = df.iloc[:, -35:]
    
    # Catrgory names for visualization purposes
    category_names = y.columns 
    
    return X, y, category_names

def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Remove user handles
    user_handle_regex = '@[^\s]+'
    detected_user_handles = re.findall(user_handle_regex, text)
    for user_handle in user_handle_regex:
        text = text.replace(user_handle, '')
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_pipeline():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
            
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def build_model():
    pipeline = build_pipeline()
    
    parameters = {
        'clf__estimator__learning_rate': [0.001, 0.01],
        'clf__estimator__n_estimators': [10, 20]
             }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model 
    
    Arguments:
        model -> Model to be evaluated
        X_test -> Test set of features
        Y_test -> Test targets
        category_names -> Target names
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=y.columns.values))


def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> Model object
        model_filepath -> Destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))


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