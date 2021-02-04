# This is not generating that many great charts. Maybe look for improvements or redo...

#  %% Setup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# %% Read Comments
events = ['2020Nov', '2020Oct', '2020Sept', '2020Jun', '2019Sept','2019Jun', '2019Mar', '2018Oct', '2018Sept', '2018Jun', '2018Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/aggregated_events/" + event + "_aggregated_Data.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% array to string 
str_comments_per_event = {}
for element in comments_per_event:
    str_comments_per_event.update(dict(zip([element], [' '.join(comments_per_event[element])])))
#    str_comments_per_event[element] = [str_comments_per_event[element]]

# %%  Filter comments and create TFIDF
# filter_words = ['nan', 'appl']
# comments_event = comments_per_event['2020Oct']
# filtered_comments_event = [' '.join([word for word in comment.split() if str(word) not in filter_words]) for comment in comments_event]
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5, max_df=1.0, norm="l2")
# X = vectorizer.fit_transform(filtered_comments_event)

string = []
count = 0
filter_words = ['nan', 'appl']

for event in str_comments_per_event:
    cleaned_comments = ' '.join([word for word in str_comments_per_event[event].split() if word not in filter_words])
    string.append(cleaned_comments)


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=1.0, norm="l2")
X = vectorizer.fit_transform(string)
print(X.shape)

# %% Do kmeans
from sklearn.cluster import KMeans

cluster = 3
km = KMeans(
    n_clusters=cluster, 
    init='k-means++', 
    max_iter=300)
km.fit(X) # km in neues dict speichern

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
labels = {
    0: 'Event for WWDC and Services',
    1: 'Event for iPad and MacBook',
    2: 'Event for iPhone and Apple Watch (TV)'
}
colors = ['r', 'g', 'b']
for group in np.unique(clusters):
    ix = np.where(clusters == group)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=labels[group], c=colors[group], s=75, alpha=1)


ax.set_title('Text Clustering of Events (Aggregated)')
ax.legend()
plt.xlabel("PCA 0")
plt.ylabel("PCA 1")
for pos, event in enumerate(str_comments_per_event.keys()):
    plt.annotate(event, (scatter_x[pos], scatter_y[pos]))
plt.show()

# %% Investigate Clusters
# Get top words for each cluster
# comments_with_clusters = list(zip(filtered_comments_event, km.labels_))
comments_with_clusters = list(zip(string, km.labels_))

comments_sorted_by_cluster = {}
for i in range(30):
    comments_sorted_by_cluster[i] = []

for comment in comments_with_clusters:
    cluster = comment[1]
    text = comment[0]
    comments_sorted_by_cluster[cluster].append(text)

comments_sorted_by_frequency = {}
for i in range(30):
    comments_sorted_by_frequency[i] = {}

for cluster in comments_sorted_by_cluster.keys():
    for comment in comments_sorted_by_cluster[cluster]:
        for word in comment.split():
            if word in comments_sorted_by_frequency[cluster]:
                comments_sorted_by_frequency[cluster][word] += 1
            else:
                comments_sorted_by_frequency[cluster][word] = 1
    comments_sorted_by_frequency[cluster] = sorted(comments_sorted_by_frequency[cluster].items(), key=lambda x: x[1], reverse=True)[:25]
# %% Print top words per cluster
for cluster in comments_sorted_by_frequency:
    print("Top words for cluster " + str(cluster))
    for word in comments_sorted_by_frequency[cluster]:
        print("  " + word[0] + ": " + str(word[1]))

# %% OPTIONAL: TSNE
from yellowbrick.text import TSNEVisualizer

tsne = TSNEVisualizer()
tsne.fit(X, km.labels_)
tsne.show()


# %%
