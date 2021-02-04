# %% Do LDA Topic Modeling
# Imports
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd

# %% Import Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/cleaned_nouns/"+event+"_CleanedNounsOnly.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% Clean NaN
for event in comments_per_event:
    filter_list = ['nan']
    cleaned_comments = [' '.join([word for word in comment.split() if str(word) not in filter_list]) for comment in comments_per_event[event]]
    comments_per_event[event] = cleaned_comments
# %% 
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
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1, max_iter=50, learning_method='online', random_state=100, )
lda.fit(doc_term_matrix)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, comments_per_event['2020_Nov_Post'], number_words)

lda.score(doc_term_matrix)
# %% Idea 1: Search optimal number of topics via GridSearch
# https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/

search_params = {'n_components': [1, 2, 3, 4], 'learning_decay': [0.3, 0.4, 0.5]}

count_vect = CountVectorizer(max_df=1.0, min_df=0.01, stop_words='english')
doc_term_matrix = count_vect.fit_transform(comments_per_event['2020_Nov_Post'])

lda = LDA(learning_method='online')
model = GridSearchCV(lda, param_grid=search_params)
model.fit(doc_term_matrix)

best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix))
# %% Idea 2: Optimal number of topics via coherence score
import gensim.corpora as corpora
import gensim
import matplotlib.pyplot as plt 

lda_events = ["2020_Nov_Pre", "2019_Mar_Pre", "2018_Sept_Pre", "2020_Nov", "2019_Sept", "2018_Oct"]
lda_data = []

for event in lda_events:
    print("Doing " + event)
    comments = [comment.split() for comment in comments_per_event[event] if len(comment.split()) > 0]
    id2word = corpora.Dictionary(comments)
    corpus = [id2word.doc2bow(comment) for comment in comments]

    # Find best topics
    topics = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    scores = []
    models = []

    for topic in topics:
        chunksize = 500 
        passes = 20 
        iterations = 400
        eval_every = 1  

        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                            alpha='auto', eta='auto', \
                            iterations=iterations, num_topics=topic, \
                            passes=passes, eval_every=eval_every)
        models.append(lda_model)

        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=comments, dictionary=id2word, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        scores.append(coherence_lda)
        # Print the Keyword in the 5 topics
        #print(lda_model.print_topics())
        print("  \___ " + str(topic) + ": " + str(coherence_lda))

    fig, ax = plt.subplots()
    ax.plot(topics, scores)
    ax.set(xlabel='Topcis', ylabel='Coherence Score (u_mass)', title='Optimal Number of Topics for ' + event)
    ax.grid()
    plt.show()
    fig.savefig("../images/topic_modeling/coherence_"+event+".pdf")

    lda_data.append([models[scores.index(min(scores))], corpus, id2word])

# %%
import pyLDAvis.gensim

lda_visualization = pyLDAvis.gensim.prepare(lda_data[5][0], lda_data[5][1], lda_data[5][2], sort_topics=True, mds="tsne")
pyLDAvis.display(lda_visualization)
# %%
