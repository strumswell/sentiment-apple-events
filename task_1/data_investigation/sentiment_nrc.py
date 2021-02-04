#  %% Imports
from nrclex import NRCLex
import pandas as pd

# %% Read Docs
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/unprocessed/" + event + "_Data.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

# %% Sentiment Analysis based off of NRC
# Vars for Plot
plt_labels = []
plt_positive = []
plt_negative = []

# Generate sentiment
for event in comments_per_event:
    #print(event)
    plt_labels.append(event)
    # Generate emotions per comments and store their result
    overall_scores = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0 ,'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    for comment in comments_per_event[event]:
        comment_emotion = NRCLex(comment)
        for emotion in comment_emotion.affect_frequencies:
            overall_scores[emotion] += comment_emotion.affect_frequencies[emotion]
    #print(overall_scores)

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

    plt_positive.append(round(positive_percent))
    plt_negative.append(round(negative_percent))
    #print('Positive: ' + str(positive_percent) + ' %\n')
    #print('Positive: ' + str(negative_percent) + ' %\n')
    #print('\n\n\n')

# %% Plotting results
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(plt_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(17.5, 10.5)
rects1 = ax.bar(x - width/2, plt_positive, width, label='Positive', color='g')
rects2 = ax.bar(x + width/2, plt_negative, width, label='Negative', color='r')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores in %')
ax.set_title('Scores by event and sentiment')
ax.set_xticks(x)
ax.set_xticklabels(plt_labels)
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

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.xticks(rotation=90)
plt.show()
######## ONLY SINGLE EVENT
# %% Generate emotions per comments and store their result
overall_scores = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0 ,'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}

for comment in comments_per_event['2020_Nov']:
    comment_emotion = NRCLex(comment)
    for emotion in comment_emotion.affect_frequencies:
        overall_scores[emotion] += comment_emotion.affect_frequencies[emotion]
        #if emotion == 'surprise' and comment_emotion.affect_frequencies[emotion] > 0.0:
        #    print(comment)
        #    print('\n')

print(overall_scores)
# %%  Sum positive and negative emotions
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

print('Positive: ' + str(positive_percent) + ' %\n')
print('Positive: ' + str(negative_percent) + ' %\n')

# %%
# %% Testing NRC out
example_comment = comments_per_event['2020_Nov_Post'][0]
emotion = NRCLex(example_comment)

print('\n', emotion.words) 
print('\n', emotion.sentences) 
print('\n', emotion.affect_list) 
print('\n', emotion.affect_dict) 
print('\n', emotion.raw_emotion_scores) 
print('\n', emotion.top_emotions) 
print('\n', emotion.affect_frequencies) 