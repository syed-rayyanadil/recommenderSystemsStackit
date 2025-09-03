from flask import Flask, request, jsonify
from preprocessing import loadAndCleanData
from model import getTfidfRecommendations, getKMeansRecommendations

app = Flask(__name__)

dfClean = loadAndCleanData('../data/25k IMDb movie Dataset.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    watched = data.get('watched_movies', [])
    if not watched or not isinstance(watched, list):
        return jsonify({'error': 'Please provide a list of watched_movies'}), 400

    tfidfRecs = getTfidfRecommendations(dfClean, watched, topN=5)
    kmeansRecs = getKMeansRecommendations(dfClean, watched, topN=5)

    tfidfList = tfidfRecs.to_dict(orient='records')
    kmeansList = kmeansRecs.to_dict(orient='records')

    return jsonify({
        'tfidf_recommendations': tfidfList,
        'kmeans_recommendations': kmeansList
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
