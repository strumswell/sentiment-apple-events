# https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/?ref=lbp

#  %% Imports
import utils.preprocessing as pre
import pandas as pd
import nltk
import inflect 
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# %% Functions
# https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/?ref=lbp
p = inflect.engine()
stemmer = SnowballStemmer('english')
lemmer=WordNetLemmatizer()

def convert_number(text): 
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 
  
    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word) 
  
    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str 

def adjust_abbreviations(comment):
    if type(comment) is not str:
        comment = str(comment)

    return comment.replace('$', ' dollar ') \
                  .replace('€', ' euro') \
                  .replace('gb', ' gb') \
                  .replace('5g', '5 g') \
                  .replace('mm ', '  mm ') \
                  .replace(' mb', ' macbook') \
                  .replace(' mbp', ' macbook pro') \
                  .replace(' mba', ' macbook air') \
                  .replace('gelöscht', '')

def get_custom_word_list():
    stop_words = nltk.corpus.stopwords.words('english')
    new_words = ['didnt', 'wouldnt', 'wasnt', 'hasnt', 'dont','doesnt', 'isnt', 'arent', 'im', 'shouldve', 'havent', 'youll' ]
    stop_words.extend(new_words)
    return stop_words

def stem_comment(comment):
    return ' '.join([stemmer.stem(word) for word in comment.split()])

def lemm_comment(comment):
    return ' '.join([lemmer.lemmatize(word) for word in comment.split()])


# %% Import Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../data/unprocessed/"+event+"_Data.csv"))

# %% Clean Comments
wpt = nltk.WordPunctTokenizer()
stop_words = get_custom_word_list()

pos = 1
for df in dfs:
    print("Doing " + str(pos) + " of " + str(len(dfs)))
    body = df['Body']
    
    # Lowercase
    body = body.str.lower()
    # Translate $-Symbol to Word
    body = [adjust_abbreviations(comment) for comment in body]

    # Filter out URLs
    body = [re.sub(r'https?://\S+', '', comment, re.I|re.A) for comment in body]

    # Number to WordNumber (6 -> six)
    body = [convert_number(comment) for comment in body]

    body_tokenized = [wpt.tokenize(comment) for comment in body]
    body_filtered_tokenized = [' '.join([token for token in comment if token not in stop_words]) for comment in body_tokenized] 

    # Remove Number + special chars
    body = [re.sub(r'[^a-zA-Z\s]', '', comment, re.I|re.A) for comment in body_filtered_tokenized]
    
    # Lemmatize 
    body = [stem_comment(comment) for comment in body]

    # Remove Whitespaces
    body = [" ".join(comment.split()) for comment in body]

    df['Body'] = body
    pos += 1
#  %% Save cleaned DataFrames
for pos, df in enumerate(dfs):
    df.to_csv("../data/cleaned/"+ events[pos] + "_CleanedData.csv")
    print("Saved " + str(pos) + " of " + str(len(dfs)))

# %%
test = dfs[0]
print(dfs[0]['Body'][448])

# %% 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import matplotlib.pyplot as plt 

documents = [[str(comment) for comment in df['Body']] for df in dfs]
for pos, doc in enumerate(documents):
    print("Doing " + str(pos))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=10, max_features=100000, stop_words='english', ngram_range=(1, 3))
    tfidf = tfidf_vectorizer.fit_transform(doc)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    num_clusters = 4
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf)
    clusters = km.labels_.tolist()

    pca = PCA(n_components=2).fit(tfidf.todense())
    data2D = pca.transform(tfidf.todense())
    plt.figure(figsize=(28,20))
    plt.scatter(data2D[:,0], data2D[:,1], c=km.labels_)
    #plt.show()
    plt.savefig('images/kmeans/' + events[pos] + ".pdf")

















#plt.savefig('images/kmeans/' + events[pos] + ".png")

# %% 
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf vectorizer of scikit learn
vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=100000, max_df = 0.5, use_idf = True, ngram_range=(1,3))
X = vectorizer.fit_transform(dfs[0]['Body'])
print(X.shape) # check shape of the document-term matrix
terms = vectorizer.get_feature_names()

#  %% 
from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(X)
clusters = km.labels_.tolist()

#  %% 
# applying lsa //////////////////////////////

from sklearn.utils.extmath import randomized_svd
U, Sigma, VT = randomized_svd(X, n_components=5, n_iter=100,
                              random_state=122)
#printing the concepts
for i, comp in enumerate(VT):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Concept "+str(i)+": ")
        for t in sorted_terms:
            print(t[0])
        print(" ")

# %% ####### LDA
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 1000

documents = dfs[0]['Body']

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

#  %% 
import umap
import matplotlib.pyplot as plt
X_topics=U*Sigma
embedding = umap.UMAP(n_neighbors=250, min_dist=0.5, random_state=12).fit_transform(X_topics)
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = clusters,
    s = 10, # size
    edgecolor='none'
)
plt.show()

# %%
from sklearn.decomposition import LatentDirichletAllocation
# Run LDA
lda = LatentDirichletAllocation(n_components=5, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

# %%

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 25
display_topics(lda, tf_feature_names, no_top_words)
# %%

# %%
