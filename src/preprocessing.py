import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler


def basicEda(df, valueCountsMaxCols=5):
    print("SHAPE:", df.shape)
    print("COLUMNS:", df.columns.tolist())
    try:
        print("Duplicated rows:", df.duplicated().sum())
    except TypeError:
        print("Skipping duplicated row check due to unhashable types in columns.")
    print("Missing values:\n", df.isnull().sum())
    print("Data types:\n", df.dtypes)
    print("Sample data:\n", df.head(2))
    print(f"Value counts (up to {valueCountsMaxCols} columns):")


def valueCounts(df, column, top=10):
    print(f"Value counts for `{column}`:")
    print(df[column].value_counts().head(top))


def removeReadAll(text):
    if isinstance(text, str) and text.endswith('Read all'):
        return text[:-8].strip()
    return text


def convertToInt(rating):
    try:
        rating = str(rating).upper().strip()
        if 'K' in rating:
            return int(float(rating.replace('K', '')) * 1_000)
        elif 'M' in rating:
            return int(float(rating.replace('M', '')) * 1_000_000)
        else:
            return int(rating)
    except:
        return np.nan


def weightedSortingScore(df, w1=48, w2=52):
    return (df["Rating_Count_Scaled"] * w1 / 100 +
            df["Rating"] * w2 / 100)


def parseListColumn(value):
    if pd.isna(value):
        return []
    try:
        return literal_eval(value)
    except:
        return []


def cleanYear(value):
    value = str(value).strip().replace('-', '')
    return int(value) if value.isdigit() else np.nan


def cleanMovies(df):
    df = df.dropna(subset=['Overview'])
    df = df[df['Overview'].str.lower() != 'none']
    df['Overview'] = df['Overview'].apply(removeReadAll)
    df['Plot Kyeword'] = df['Plot Kyeword'].apply(parseListColumn)
    df['Description'] = df['Overview'] + ' ' + df['Plot Kyeword'].apply(lambda x: ' '.join(x))
    df['word_count'] = df['Description'].str.split().str.len()
    df = df[df['word_count'] > 15]
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['User Rating'] = df['User Rating'].apply(convertToInt)
    df = df[df['User Rating'] > 0]
    df = df.dropna(subset=['Rating'])
    scaler = MinMaxScaler(feature_range=(1, 10))
    df['Rating_Count_Scaled'] = scaler.fit_transform(df[['User Rating']])
    df['Weighted_Rating'] = weightedSortingScore(df)
    df['year'] = df['year'].apply(cleanYear)

    if 'Generes' in df.columns:
        df = df.rename(columns={'Generes': 'Genres'})

    if 'Plot Kyeword' in df.columns:
        df = df.rename(columns={'Plot Kyeword': 'Plot Keyword'})
    columnsToDrop = ["Run Time", "User Rating", "word_count"]
    df = df.drop(columns=[col for col in columnsToDrop if col in df.columns])
    df = df.sort_values(by="Weighted_Rating", ascending=False).reset_index(drop=True)

    return df


def loadAndCleanData(filepath):
    df_raw = pd.read_csv(filepath)
    basicEda(df_raw)
    df_clean = cleanMovies(df_raw)
    basicEda(df_clean)
    df_clean.to_csv('../data/cleaned_movies.csv', index=False)
    return df_clean