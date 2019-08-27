import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Valores nulos - no hay
    
    # Datos categóricos: Sex & Deck
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    df['smoker'] = LabelEncoder().fit_transform(df['smoker'])
    df['region'] = LabelEncoder().fit_transform(df['region'])
    
    return df