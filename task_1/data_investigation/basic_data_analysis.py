#  %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# %% Read Docs
events = list(reversed(['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']))
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/unprocessed/" + event + "_Data.csv"))

comments_per_event = dict(zip(events, [[str(comment) for comment in df['Body'].values] for df in dfs]))

#  %% Plot Comment, Author and Score Distribution
counts = [df.describe()['Score']['count'] for df in dfs]
min_scores = [min(list(df['Score'])) for df in dfs]
max_scores = [max(list(df['Score'])) for df in dfs]
mean_scores = [round(df.describe()['Score']['mean'], 2) for df in dfs]
unique_authors = [df['Author'].nunique() for df in dfs]
comment_per_author = [round(df.describe()['Score']['count']/df['Author'].nunique(), 2) for df in dfs]

x = np.arange(len(events))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
fig.suptitle('Comment, Author and Score Distribution')
fig.set_size_inches(15, 12)

ax1.bar(events, counts, label="Comments", color='b')
ax1.set_ylabel('Count')
ax1.bar(events, unique_authors, label="Unique Authors", color='orange')
ax1.tick_params(labelrotation=90)
ax1.legend()

ax2.plot(max_scores, color="g", marker="o")
ax2.set_ylabel('Max Scores', color="g")

ax2t = ax2.twinx()
ax2t.plot(min_scores, color="r", marker="o")
ax2t.set_ylabel("Min Scores", color="red")

ax2.set_xticks(x)
ax2.set_xticklabels(events)
ax2.tick_params(labelrotation=90)


fig.tight_layout()
plt.show()

# %% Plot Aggregated Mean Scores
pre = [[i, e] for i, e in enumerate(events) if 'Pre' in e]
post = [[i, e] for i, e in enumerate(events) if 'Post' in e]
dur = [[i, e] for i, e in enumerate(events) if 'Pre' not in e and 'Post' not in e]

pre_counts_mean = round(sum([counts[event[0]] for event in pre]) / len(pre))
post_counts_mean = round(sum([counts[event[0]] for event in post]) / len(post))
dur_counts_mean = round(sum([counts[event[0]] for event in dur]) / len(dur))
mean_count = [pre_counts_mean, dur_counts_mean, post_counts_mean]

pre_score_max_mean = round(sum([max_scores[event[0]] for event in pre]) / len(pre))
post_score_max_mean = round(sum([max_scores[event[0]] for event in post]) / len(post))
dur_score_max_mean = round(sum([max_scores[event[0]] for event in dur]) / len(dur))
mean_max = [pre_score_max_mean, dur_score_max_mean, post_score_max_mean]

pre_score_min_mean = round(sum([min_scores[event[0]] for event in pre]) / len(pre))
post_score_min_mean = round(sum([min_scores[event[0]] for event in post]) / len(post))
dur_score_min_mean = round(sum([min_scores[event[0]] for event in dur]) / len(dur))
mean_min = [pre_score_min_mean, dur_score_min_mean, post_score_min_mean]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
fig.set_size_inches(10, 7)
fig.suptitle('Mean Comment and Score Distribution')

x = np.arange(len(events))
ax1.bar(['Pre-Event', 'Event', 'Post-Event'], mean_count, color='b')
ax1.set_ylabel('Mean Comment Count', color="b")

ax2.bar(['Pre-Event', 'Event', 'Post-Event'], mean_max, color='g')
ax2.set_ylabel('Mean Max Score', color="g")

ax3.bar(['Pre-Event', 'Event', 'Post-Event'], mean_min, color='r')
ax3.set_ylabel('Mean Min Score', color="r")

fig.tight_layout()
plt.show()

# %% Plot Upvote Proportions
filtered_scores_50 = [len(df.index[df['Score'] > 50]) for df in dfs]
filtered_scores_20 = [len(df.index[df['Score'] > 20]) for df in dfs]
filtered_scores_10 = [len(df.index[df['Score'] > 10]) for df in dfs]

fig, ax = plt.subplots()
fig.suptitle('Upvoted Comments Proportion')
fig.set_size_inches(12, 8)

x = np.arange(len(events))
ax.bar(events, counts, label="All Comments", color='gray')
ax.bar(events, filtered_scores_10, label="More than 10 Upvotes", color='red')
ax.bar(events, filtered_scores_20, label="More than 20 Upvotes", color='orange')
ax.bar(events, filtered_scores_50, label="More than 50 Upvotes", color='yellow')
ax.set_ylabel('Comments')
ax.legend()

ax.set_xticks(x)
ax.set_xticklabels(events)
ax.tick_params(labelrotation=90)

fig.tight_layout()
plt.show()
# %%
