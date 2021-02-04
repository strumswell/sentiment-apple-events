#  %% Import
import nltk 
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# %% Import Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/cleaned_nouns/"+event+"_CleanedNounsOnly.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% Combine comments per Event
str_comments_per_event = [' '.join([' '.join(comments_per_event[event])][0].split()) for event in comments_per_event]

# %% TFIDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_df=1.0, norm="l2")
X = vectorizer.fit_transform(str_comments_per_event)
print(X.shape)

# %% Do kmeans
from sklearn.cluster import KMeans

cluster = 4
km = KMeans(
    n_clusters=cluster,
    init='k-means++',
    max_iter=1000000)
km.fit(X) # km in neues dict speichern

# %% Do PCA to reduce dimensions
from sklearn.decomposition import PCA

clusters = km.predict(X.todense())

pca = PCA(n_components=2)
two_dim = pca.fit_transform(X.todense())

scatter_x = two_dim[:, 0] # first principle component
scatter_y = two_dim[:, 1] # second principle component

# %% Plot dim reduction results
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig, ax = plt.subplots()
fig.set_size_inches(20,10)
ax.set_title('Text Clustering of Events')

colors = ['r', 'g', 'b', 'y']
labels = ['WWDC', 'iPhone & Watch Event', 'Services Event', 'iPad & MacBook Events']
for group in np.unique(clusters):
    ix = np.where(clusters == group)
    ax.scatter(scatter_x[ix], scatter_y[ix], label=labels[group], c=colors[group], s=75, alpha=1)

for i, event in enumerate(events):
    ax.annotate(event, (scatter_x[i], scatter_y[i]))

ax.legend()

plt.xlabel("Component 0")
plt.ylabel("Component 1")
plt.show()

#  %% Save Clusters to disk
cluster_data = [[event, clusters[i], labels[clusters[i]]] for i, event in enumerate(events)]
cluster_dataframe = pd.DataFrame(cluster_data, columns =['Event', 'Cluster', 'Cluster Label'])
cluster_dataframe.to_csv("images/document_clustering/cluster_data.csv", index = False, header=True)

# %% Investigate Clusters
# Get top words for each cluster
# comments_with_clusters = list(zip(filtered_comments_event, km.labels_))
comments_with_clusters = list(zip(str_comments_per_event, km.labels_))

comments_sorted_by_cluster = {}
for i in range(4):
    comments_sorted_by_cluster[i] = []

for comment in comments_with_clusters:
    cluster = comment[1]
    text = comment[0]
    comments_sorted_by_cluster[cluster].append(text)

comments_sorted_by_frequency = {}
for i in range(4):
    comments_sorted_by_frequency[i] = {}

for cluster in comments_sorted_by_cluster.keys():
    for comment in comments_sorted_by_cluster[cluster]:
        for word in comment.split():
            if word in comments_sorted_by_frequency[cluster]:
                comments_sorted_by_frequency[cluster][word] += 1
            else:
                comments_sorted_by_frequency[cluster][word] = 1
    comments_sorted_by_frequency[cluster] = sorted(comments_sorted_by_frequency[cluster].items(), key=lambda x: x[1], reverse=True)[:50]
# %% Print top words per cluster
for cluster in comments_sorted_by_frequency:
    print("Top words for cluster " + str(cluster))
    for word in comments_sorted_by_frequency[cluster]:
        print("  " + word[0] + ": " + str(word[1]))

# %% Event Frequency per Month
months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
colors = ['r', 'g', 'b', 'y']

freq_0 = [[events[i].split("_")[1] for i, event in enumerate(clusters) if event == 0].count(month) for month in months]
freq_1 = [[events[i].split("_")[1] for i, event in enumerate(clusters) if event == 1].count(month) for month in months]
freq_2 = [[events[i].split("_")[1] for i, event in enumerate(clusters) if event == 2].count(month) for month in months]
freq_3 = [[events[i].split("_")[1] for i, event in enumerate(clusters) if event == 3].count(month) for month in months]

x = np.arange(len(months))  # the label locations
width = 0.15  # the width of the bars

r0 = np.arange(len(freq_0))
r1 = [x + width for x in r0]
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

fig, ax = plt.subplots()
fig.set_size_inches(12.5, 7.5)
rects0 = ax.bar(r0, freq_0, width, color=colors[0], label='WWDC')
rects1 = ax.bar(r1, freq_1, width, color=colors[1], label='iPhone & Watch Events')
rects2 = ax.bar(r2, freq_2, width, color=colors[2], label='Services Event')
rects3 = ax.bar(r3, freq_3, width, color=colors[3], label='iPad & MacBook Events')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency')
ax.set_xlabel('Month')
ax.set_title('Event Cluster Frequency per Month')
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

fig.tight_layout()

plt.show()
# %% WordClouds for Clusters
from wordcloud import WordCloud

for cluster in comments_sorted_by_frequency:
    print("Doing " + str(cluster))
    wordcloud = WordCloud(background_color="white", max_words=50, contour_width=3, contour_color='steelblue', width=1920, height=1080).generate_from_frequencies(dict(comments_sorted_by_frequency[cluster]))
    wordcloud.to_image()
    wordcloud.to_file("images/document_clustering/cluster_" + str(cluster) + "_wordcloud.png")

# %% Do LDA Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=1.0, min_df=0.01, stop_words='english')
doc_term_matrix = count_vect.fit_transform(comments_per_event['2020_Nov_Post'])

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vect.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 7
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1, max_iter=50, learning_method='online', random_state=100, )
lda.fit(doc_term_matrix)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, comments_per_event['2020_Nov_Post'], number_words)
# %%
