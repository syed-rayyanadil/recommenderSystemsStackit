import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def getTfidfRecommendations(df, movieTitles, topN=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidfMatrix = tfidf.fit_transform(df['Description'])

    indices = pd.Series(df.index, index=df['movie title']).drop_duplicates()

    validIndices = [indices[title] for title in movieTitles if title in indices]
    if not validIndices:
        print("No valid movies found in input.")
        return pd.DataFrame()

    simMatrix = cosine_similarity(tfidfMatrix[validIndices], tfidfMatrix)
    avgSimScores = simMatrix.mean(axis=0)

    similarIndices = avgSimScores.argsort()[::-1]
    similarIndices = [i for i in similarIndices if i not in validIndices]

    topIndices = similarIndices[:topN]
    return df.iloc[topIndices][['movie title', 'Weighted_Rating']]

def getKMeansRecommendations(df, movieTitles, numClusters=20, topN=5):
    descVectors = TfidfVectorizer(stop_words='english').fit_transform(df['Description'])
    kmeans = KMeans(n_clusters=numClusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(descVectors)

    indices = pd.Series(df.index, index=df['movie title']).drop_duplicates()
    validIndices = [indices[title] for title in movieTitles if title in indices]
    if not validIndices:
        print("No valid movies found in input.")
        return pd.DataFrame()

    selectedClusters = df.loc[validIndices, 'Cluster']
    commonCluster = selectedClusters.mode().iloc[0]

    clusterMovies = df[(df['Cluster'] == commonCluster) & (~df.index.isin(validIndices))]
    return clusterMovies.sort_values(by='Weighted_Rating', ascending=False).head(topN)[['movie title', 'Weighted_Rating']]
