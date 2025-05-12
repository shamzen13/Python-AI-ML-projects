from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    points = data['points']
    cluster_centers = data['clusterCenters']

    if len(cluster_centers) == 0:
        return jsonify({'error': 'No cluster centers provided'}), 400

    coords = np.array([[p['x'], p['y']] for p in points])
    init_centers = np.array([[c['x'], c['y']] for c in cluster_centers])

    kmeans = KMeans(n_clusters=len(init_centers), init=init_centers, n_init=1, max_iter=300).fit(coords)
    labels = kmeans.labels_.tolist()

    return jsonify({'assignments': labels})


if __name__ == '__main__':
    app.run(debug=True)
