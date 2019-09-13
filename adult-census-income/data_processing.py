import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_adult_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Valores nulos
    
    
    # Datos categóricos:
    df['workclass'] = LabelEncoder().fit_transform(df['workclass'])
    df['education'] = LabelEncoder().fit_transform(df['education'])
    df['marital-status'] = LabelEncoder().fit_transform(df['marital-status'])
    df['occupation'] = LabelEncoder().fit_transform(df['occupation'])
    df['relationship'] = LabelEncoder().fit_transform(df['relationship'])
    df['country'] = LabelEncoder().fit_transform(df['country'])
    df['salary'] = LabelEncoder().fit_transform(df['salary'])
    df['occupation'] = LabelEncoder().fit_transform(df['occupation'])
    df['race'] = LabelEncoder().fit_transform(df['race'])
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    
    # Selección de variables
    #df = df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], 1,)
    
    return df