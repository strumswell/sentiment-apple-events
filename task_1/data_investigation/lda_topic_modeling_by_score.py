# %% Do LDA Topic Modeling
# Imports
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd

# %% Import Cleaned Documents
events = ["2020_Nov_Pre", "2019_Mar_Pre", "2018_Sept_Pre", "2020_Nov", "2018_Oct"]
dfs = []
score = "25"

for event in events:
    dfs.append(pd.read_csv("../../data/cleaned/"+event+"_CleanedData.csv"))

comments_per_event = dict(zip(events, [[str(comment) for i, comment in enumerate(df['Body'].values) if df['Score'][i] >= int(score)] for df in dfs]))

# %% Save uncleaned negative comments to json file
import json

uncleaned_dfs = []
events = ["2020_Nov_Pre", "2019_Mar_Pre", "2018_Sept_Pre", "2020_Nov", "2018_Oct"]

for event in events:
    uncleaned_dfs.append(pd.read_csv("../../data/unprocessed/"+event+"_Data.csv"))

uncleaned_comments_per_event = dict(zip(events, [[str(comment) for i, comment in enumerate(df['Body'].values) if df['Score'][i] >= int(score)] for df in uncleaned_dfs]))


with open('../../data/unprocessed/score/comments_'+score+'.json', 'w') as fp:
    json.dump(uncleaned_comments_per_event, fp, indent=4)

# %% Remove custom filter words
for event in comments_per_event:
    filter_list = ['nan', 'appl', 'entfernt']
    cleaned_comments = [' '.join([word for word in comment.split() if str(word) not in filter_list]) for comment in comments_per_event[event]]
    comments_per_event[event] = cleaned_comments

# %% Prepare lda dict
lda_data = {}

# %% Find optimal Topic number via Coherence Scores
import gensim.corpora as corpora
import gensim
import matplotlib.pyplot as plt 

lda_events = ["2020_Nov_Pre", "2019_Mar_Pre", "2018_Sept_Pre", "2020_Nov", "2018_Oct"]

for event in lda_events:
    # Results for specific topic
    result_scores = [] # coherence scores for different lda models 
    result_models = [] # different lda models (per topic nr)

    print("Doing " + event)
    comments = [[word for word in comment.split() if len(word) > 1] for comment in comments_per_event[event]]
    id2word = corpora.Dictionary(comments)
    corpus = [id2word.doc2bow(comment) for comment in comments]
    topics = [*range(2, 11)] # 2 ... 10

    # Create lda model for each number of topics
    for topic in topics:
        passes = 20 
        iterations = 100
        eval_every = 1  

        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, \
                            alpha='auto', eta='auto', \
                            iterations=iterations, num_topics=topic, \
                            passes=passes, eval_every=eval_every)
        result_models.append(lda_model)

        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        
        result_scores.append(coherence_lda)
        print("  |___ " + str(topic) + ": " + str(coherence_lda))

    # Keep results for event
    lda_data[event] = {
        "topics": topics, 
        "scores": result_scores, 
        "models": result_models, 
        "corpus": corpus, 
        "id2word": id2word}

# %% Plot Coherence Scores per Event
fig, axes = plt.subplots(3, 2, figsize=(10,10))
fig.suptitle('Coherence Scores for Prominent Events')

for i, event in enumerate(lda_data):
    ax = axes.flatten()[i]
    fig.add_subplot(ax)
    ax.set(xlabel='Topics', ylabel='Coherence Score (u_mass)', title=event)
    ax.plot(range(2, 11), lda_data[event]['scores'])
    
plt.tight_layout()
#plt.show()
plt.savefig('../images/topic_modeling/score/coherence_score_'+score+'.pdf')

# %% Generate WordCloud per Topic
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#6.-What-is-the-Dominant-topic-and-its-percentage-contribution-in-each-document

from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=15,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

for event in lda_data:
    # Find best lda model (min coherence score)
    best_model_index = lda_data[event]['scores'].index(min(lda_data[event]['scores']))
    lda_model = lda_data[event]['models'][best_model_index]

    # Get 15 most frequent words of topics
    topics = lda_model.show_topics(num_words=15, formatted=False)

    fig, axes = plt.subplots(4, 3, figsize=(10,10), sharex=True, sharey=True)
    fig.suptitle('Topic WordClouds for ' + event )

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        if (i <= best_model_index + 1):
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig('../images/topic_modeling/score/topic_clouds_score_' + score + '_' + event + ".pdf")

# %% Investigation of Topics via pyLDAvis (optional)
import pyLDAvis.gensim

event = '2018_Sept_Pre'
best_model_index = lda_data[event]['scores'].index(min(lda_data[event]['scores']))
lda_model = lda_data[event]['models'][best_model_index]

lda_visualization = pyLDAvis.gensim.prepare(lda_model, lda_data[event]['corpus'], lda_data[event]['id2word'], sort_topics=True)
pyLDAvis.display(lda_visualization)

############### Sentiment Analysis of each Topic ###############
# %% Import uncleaned Documents
dfs_uncleaned = []

for event in events:
    dfs_uncleaned.append(pd.read_csv("../../data/unprocessed/"+event+"_Data.csv"))

comments_per_event_uncleaned = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs_uncleaned]))

# %% Sort comments into correct topics 
ordered_documents = {}
for event in lda_data:
    document_topics = lda_model.get_document_topics(lda_data[event]['corpus'])
    ordered_document = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    best_model_index = lda_data[event]['scores'].index(min(lda_data[event]['scores']))
    lda_model = lda_data[event]['models'][best_model_index]

    for i, comment_ratings in enumerate(lda_model.get_document_topics(lda_data[event]['corpus'])):
        for rating in comment_ratings:
            # Only save ratings of over 30% certainty that this comment belongs to this topic
            if rating[1] > 0.3:
                # Save index of comment
                ordered_document[rating[0]].append(i)

    ordered_documents[event] = ordered_document

# %% Generate sentiments 
from nrclex import NRCLex

sentiment_results = {}

for event in ordered_documents:
    print(event)
    topics_positive_scores = []
    topics_negative_scores = []
    topic_labels = []
    for topic in ordered_documents[event]:
        if (len(ordered_documents[event][topic]) < 1): continue
        topic_labels.append(topic + 1)

        # Generate emotions per comments and store their result
        overall_scores = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0 ,'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
        # Use index to find corresponding uncleand comment
        for commend_id in ordered_documents[event][topic]:
            comment = comments_per_event_uncleaned[event][commend_id]
            comment_emotion = NRCLex(comment)
            for emotion in comment_emotion.affect_frequencies:
                overall_scores[emotion] += comment_emotion.affect_frequencies[emotion]

        #Sum positive and negative emotions
        emotion_types = {
        'positive': ['anticipation', 'trust', 'surprise', 'positive', 'joy'], 
        'negative': ['fear', 'anger', 'negative', 'sadness', 'disgust']
        }
        emotion_scores = {'positive': 0.0, 'negative': 0.0}

        for emotion_type in emotion_types:
            emotions = emotion_types[emotion_type]
            for emotion in emotions:
                emotion_scores[emotion_type] += overall_scores[emotion]

        base = 100 / (emotion_scores['positive'] + emotion_scores['negative'])
        positive_percent = base * emotion_scores['positive']
        negative_percent = base * emotion_scores['negative']
        
        topics_positive_scores.append(round(positive_percent))
        topics_negative_scores.append(round(negative_percent))
    
    sentiment_results[event] = [topic_labels, topics_positive_scores, topics_negative_scores]
    

# %% Plotting results of sentiment analysis
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), sharex=False, sharey=False)
fig.set_size_inches(17.5, 10.5)
fig.suptitle('Sentiment Distribution per Topic and Event')

handles, labels = (0,0)

for i, ax in enumerate(axes.flatten()):
    event = list(sentiment_results.keys())[i]
    x = np.arange(len(sentiment_results[event][0]))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, sentiment_results[event][1], width, label='Positive', color='g')
    rects2 = ax.bar(x + width/2, sentiment_results[event][2], width, label='Negative', color='r')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores in %')
    ax.set_xlabel('Topic')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_results[event][0])
    ax.set_title(event)
    handles, labels = ax.get_legend_handles_labels()

    autolabel(rects1)
    autolabel(rects2)

fig.tight_layout()
fig.legend(handles=handles, loc='upper right')

plt.show()
#plt.savefig("topic_sentiments.pdf")
 # %%
