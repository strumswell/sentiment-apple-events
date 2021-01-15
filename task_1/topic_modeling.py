# Sources
#   - https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# Not so happy with found topics -> but we have to look deeper into the topics and if we can label them
# 
# %% Importing modules
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#  %% Read Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../data/cleaned/"+event+"_CleanedData.csv"))

comments_per_event = [[str(comment) for comment in df['Body'].values] for df in dfs]

# %% Most Common Words for every Event
def plot_25_most_common_words(count_data, count_vectorizer, event_name):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:25]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(25, 15/1.6180))
    plt.subplot(title='50 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    #plt.savefig('images/25_common_words_' + event_name + '.pdf')

# Initialise the count vectorizer with the English stop words
count_vectorizer_events = []
count_data_events = []

for pos, event_comments in enumerate(comments_per_event):
    # Fit and transform the processed titles
    vectorizer = CountVectorizer(stop_words='english', min_df=10, )
    count_data = vectorizer.fit_transform(comments_per_event[pos])
    count_data_events.append(count_data)
    count_vectorizer_events.append(vectorizer)
    # Visualise the 25 most common words
    #plot_25_most_common_words(count_data, count_vectorizer, events[pos])

# %% Do LDA Topic Modeling
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 20
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1, max_iter=30, learning_method='online', random_state=100, )
lda.fit(count_data[0])
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer_events[0], number_words)
# %%
