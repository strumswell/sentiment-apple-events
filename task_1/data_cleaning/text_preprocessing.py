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
    dfs.append(pd.read_csv("../../data/unprocessed/"+event+"_Data.csv"))

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
    
    # Stem 
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