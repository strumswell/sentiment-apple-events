# %% Importing modules
import pandas as pd
from wordcloud import WordCloud

#  %% Read Documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../data/cleaned/"+event+"_CleanedData.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% Function for WordCloud
def generate_wordcloud(comments, filter_words, allow_collocations, filename):
    all_comments_filtered = ' '.join([word for word in comments.split() if word not in filter_words])
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue', width=1920, height=1080, collocations=allow_collocations)
    # Generate a word cloud
    wordcloud.generate(all_comments_filtered)
    # Visualize the word cloud
    wordcloud.to_image()
    wordcloud.to_file("images/" + str(filename) + ".png")

#  %% WordCloud of all Events 
filter_words = ['nan']
all_comments = ""
for df in dfs:
    all_comments += ' '.join([str(comment) for comment in df['Body'].values])

generate_wordcloud(all_comments, filter_words, False, "wc_all_comments_without_collocations")
# %% WordCloud for each Event
filter_words = ['nan']
for event in comments_per_event:
    all_comments = ""
    all_comments += ' '.join([str(comment) for comment in comments_per_event[event]])
    generate_wordcloud(all_comments, filter_words, True, "wordclouds/wc_" + event)
    print("Finished WordCloud for " + event)
# %%
