# This is not generating that many great charts. Maybe look for improvements or redo...

#  %% Setup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# %% Read Comments
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../data/cleaned/"+event+"_CleanedData.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% 
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 2), min_df=10, max_df=0.9, norm="l2")
X = vectorizer.fit_transform(comments_per_event['2020_Nov_Pre'])

# %% Do kmeans
from sklearn.cluster import KMeans

NUMBER_OF_CLUSTERS = 3
km = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS, 
    init='k-means++', 
    max_iter=10000)
km.fit(X)

# %% Do PCA to reduce dimensions
from sklearn.decomposition import PCA

clusters = km.predict(X)

pca = PCA(n_components=2)
two_dim = pca.fit_transform(X.todense())

scatter_x = two_dim[:, 0] # first principle component
scatter_y = two_dim[:, 1] # second principle component

# %% Plot PCA results
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig, ax = plt.subplots()
fig.set_size_inches(20,10)

for group in np.unique(clusters):
    ix = np.where(clusters == group)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=group)

ax.legend()
plt.xlabel("PCA 0")
plt.ylabel("PCA 1")
plt.show()

#  %% See top terms; Caution: Not aggregated and sorted by frequency -> todo?.
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(3):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])