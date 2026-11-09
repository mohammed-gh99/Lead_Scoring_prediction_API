

import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data():
    data_url = 'https://raw.githubusercontent.com/mohammed-gh99/Lead_Scoring_prediction_API/refs/heads/master/data-week-3-lead.csv'
    df = pd.read_csv(data_url)
    df.columns = df.columns.str.lower().str.replace(' ' , '_')
    categorical_columns  = list(df.dtypes[df.dtypes == 'str'].index)
    for c in categorical_columns:
        df[c]=df[c].str.lower().str.replace(' ' , '_')
    df.annual_income = df.annual_income.fillna(df['annual_income'].median())
    df.lead_score = df.lead_score.fillna(df['lead_score'].mean())
    df['lead_source'] = df['lead_source'].fillna(df['lead_source'].mode()[0])
    df['industry'] = df['industry'].fillna(df['industry'].mode()[0])
    df['employment_status'] = df['employment_status'].fillna(df['employment_status'].mode()[0])
    df['location'] = df['location'].fillna(df['location'].mode()[0])

    return df


def train_model(df):
    numerical = ['annual_income' , 'number_of_courses_viewed' , 'interaction_count' , 'lead_score']
    categorical = [
        'lead_source' , 
        'industry' ,
        'employment_status' ,
        'location',
    ]
    pipeline = make_pipeline(
        DictVectorizer(sparse = False),
        LogisticRegression(solver='liblinear', C=5)
    )

    train_dict = df[categorical + numerical].to_dict(orient='records')
    y_train = df.converted
    
    pipeline.fit(train_dict, y_train)

    return pipeline



def save_model(filename , model):
    with open(filename , 'wb') as f_out:
        pickle.dump(model , f_out)
        print(f'model saved to {filename}')


df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)





