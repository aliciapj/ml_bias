import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Valores nulos
    
    # Embarked: Como solo tiene 2 valores nulos, los rellenaremos con el más común
    df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df)
    
    # Age: En este caso crearemos una matriz que contenga números aleatorios, que se 
    #      calculen en función del valor de la media de la edad y la desviación estándar
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = df["Age"].isnull().sum()

    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age

    df["Age"] = age_slice
    df["Age"] = df["Age"].astype(int)

    # Cabin: empieza por una letra que, investigando, representa la cubierta en la que se
    # alojaban los pasajeros. Como puede ser interesante, podemos quedarnos solo con la letra 
    # y rellenar con otra letra inventada los valores que faltan para quitarnos los nulos. 
    # Después podemos borrar el feature Cabin ya que es redundante con Deck
    df['Deck'] = df['Cabin'].fillna("U").map(lambda x: x[0])
    df = df.drop(['Cabin'], axis=1)
    
    # Datos categóricos: Sex & Deck
    df['Deck'] = LabelEncoder().fit_transform(df['Deck'])
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    
    # Selección de variables
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], 1,)
    
    return df