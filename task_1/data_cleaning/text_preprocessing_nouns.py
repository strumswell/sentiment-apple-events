#  %% Import
import nltk 
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer

# %% Import Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/unprocessed/"+event+"_Data.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% Only leave Nouns - PoS
comments_per_event_nouns = {}

for event in comments_per_event:
    print('Doing ' + event)
    comments_per_event_nouns[event] = []
    for comment in comments_per_event[event]:
        tokens = nltk.word_tokenize(comment)
        tagged = nltk.pos_tag(tokens)
        noun_sentence = ' '.join([touple[0].lower() for touple in tagged if touple[1] == 'NN' or touple[1] == 'NNS' or touple[1] == 'NNP' or touple[1] == 'NNPS' or touple[1] == 'JJ'])
        comments_per_event_nouns[event].append(noun_sentence)

# %% Do more cleaning
def get_custom_word_list():
    stop_words = nltk.corpus.stopwords.words('english')
    new_words = ['wait', 'hope', 'new', 'good','lol','f', 'e', 'gelscht', 'didnt', 'wouldnt', 'wasnt', 'hasnt', 'dont','doesnt', 'isnt', 'arent', 'im', 'shouldve', 'havent', 'youll' ]
    stop_words.extend(new_words)
    return stop_words

def stem_comment(comment):
    stemmer = SnowballStemmer('english')
    return ' '.join([stemmer.stem(word) for word in comment.split()])

def adjust_abbreviations(comment):
    if type(comment) is not str:
        comment = str(comment)

    return comment.replace(' mbp', ' macbook pro') \
                  .replace(' mba', ' macbook air') \
                  .replace('gelöscht', '') \
                  .replace('nan', '') \
                  .replace('appl', '') \

wpt = nltk.WordPunctTokenizer()
comments_per_event_cleaned = {}
stop_words = get_custom_word_list()

for event in comments_per_event_nouns:
    print('Doing ' + event)
    comments_per_event_cleaned[event] = []
    no_links = [re.sub(r'https? //\S+', '', comment, re.I|re.A) for comment in comments_per_event_nouns[event]]
    adjusted_abbr = [adjust_abbreviations(comment) for comment in no_links]
    no_special_chars = [re.sub(r'[^a-zA-Z\s]', '', comment, re.I|re.A) for comment in adjusted_abbr]
    no_special_chars_tokenized = [wpt.tokenize(comment) for comment in no_special_chars]
    no_stopwords = [' '.join([token for token in comment if token not in stop_words]) for comment in no_special_chars_tokenized] 
    stemmed_comments = [stem_comment(comment) for comment in no_stopwords]
    no_whitespaces = [" ".join(comment.split()) for comment in stemmed_comments]

    comments_per_event_cleaned[event] = no_whitespaces

# %% Save cleaned noun data
save = False
if (save):
    for i, event in enumerate(comments_per_event_cleaned):
        clean_comments = comments_per_event_cleaned[event]
        cleaned_df = dfs[i]
        cleaned_df['Body'] = clean_comments
        #cleaned_nouns_data = [[event, comments_per_event_cleaned[event]] for event in comments_per_event_cleaned]
        #cluster_dataframe = pd.DataFrame(cluster_data, columns =['Event', 'Cluster', 'Cluster Label'])
        cleaned_df.to_csv("../data/cleaned_nouns/" + event + "_CleanedNounsOnly.csv", index = False, header=True)
#  %% 