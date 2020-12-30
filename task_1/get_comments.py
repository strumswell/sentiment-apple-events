#  %% Import dependencies
import praw
import pandas as pd
import config

#  %% Connect with Reddit
reddit = praw.Reddit(
    user_agent=config.user_agent,
    client_id=config.client_id,
    client_secret=config.client_secret,
    username=config.username,
    password=config.password,
)

#  %% Get data
event_links = {
    "2020_Nov_Post": "https://www.reddit.com/r/apple/comments/jrqtg2/apples_one_more_thing_postevent_megathread/",
    "2020_Nov": "https://www.reddit.com/r/apple/comments/jrpm7i/apples_one_more_thing_event_megathread/",
    "2020_Nov_Pre": "https://www.reddit.com/r/apple/comments/jrerd9/apples_one_more_thing_preevent_megathread/",
    "2020_Oct_Post": "https://www.reddit.com/r/apple/comments/jaiwph/apples_hi_speed_post_event_megathread/",
    "2020_Oct": "https://www.reddit.com/r/apple/comments/jahf3w/apples_hi_speed_event_megathread/",
    "2020_Oct_Pre": "https://www.reddit.com/r/apple/comments/ja64z0/apples_hi_speed_preevent_megathread/",
    "2020_Sept_Post": "https://www.reddit.com/r/apple/comments/ite7j6/apples_time_flies_post_event_megathread/",
    "2020_Sept": "https://www.reddit.com/r/apple/comments/itcnxd/apples_time_flies_event_megathread/",
    "2020_Sept_Pre": "https://www.reddit.com/r/apple/comments/it23hj/apples_time_flies_preevent_megathread/",
    "2020_Jun_Post": "https://www.reddit.com/r/apple/comments/hdyaeh/wwdc_2020_postevent_megathread/",
    "2020-Jun": "https://www.reddit.com/r/apple/comments/hdw3dw/wwdc_2020_event_megathread/",
    "2020_Jun_Pre": "https://www.reddit.com/r/apple/comments/hdigku/wwdc_2020_preevent_megathread/",
    "2019_Sept_Post": "https://www.reddit.com/r/apple/comments/d2cglb/apples_by_innovation_only_postevent_megathread/",
    "2019_Sept": "https://www.reddit.com/r/apple/comments/d2af3a/apples_by_innovation_only_event_megathread/",
    "2019_Sept_Pre": "https://www.reddit.com/r/apple/comments/d2103g/apples_by_innovation_only_preevent_megathread/",
    "2019_Jun": "https://www.reddit.com/r/apple/comments/bwd0uy/wwdc_2019_event_megathread/",
    "2019_Jun_Pre": "https://www.reddit.com/r/apple/comments/bw6ekc/wwdc_2019_preevent_megathread/",
    "2019_Mar_Post": "https://www.reddit.com/r/apple/comments/b5epdq/apples_its_show_time_postevent_megathread/",
    "2019_Mar": "https://www.reddit.com/r/apple/comments/b5d6fv/apples_its_show_time_event_megathread/",
    "2019_Mar_Pre": "https://www.reddit.com/r/apple/comments/b564ux/apples_its_show_time_preevent_megathread/",
    "2018_Oct_Post": "https://www.reddit.com/r/apple/comments/9sp2ka/apples_there_is_more_in_the_making_postevent/",
    "2018_Oct": "https://www.reddit.com/r/apple/comments/9so41z/apples_there_is_more_in_the_making_event/",
    "2018_Oct_Pre": "https://www.reddit.com/r/apple/comments/9sndgm/apples_there_is_more_in_the_making_preevent/",
    "2018_Sept_Post": "https://www.reddit.com/r/apple/comments/9fain3/apples_gather_round_postevent_megathread/",
    "2018_Sept": "https://www.reddit.com/r/apple/comments/9f9b31/apples_gather_round_event_megathread/",
    "2018_Sept_Pre": "https://www.reddit.com/r/apple/comments/9f4ge7/apples_gather_round_preevent_megathread/",
    "2018_Jun_Post": "https://www.reddit.com/r/apple/comments/8ok1pj/wwdc_2018_postevent_megathread/",
    "2018_Jun": "https://www.reddit.com/r/apple/comments/8ohxfy/wwdc_2018_event_megathread/",
    "2018_Jun_Pre": "https://www.reddit.com/r/apple/comments/8oeiq0/wwdc_2018_preevent_megathread/",
    "2018_Mar": "https://www.reddit.com/r/apple/comments/87ioa4/apples_lets_take_a_field_trip_event_megathread/",
}
for event in event_links:
    print("Doing event: " + event)
    # TODO: Get general submission infos (num_comments, ...)
    submission = reddit.submission(url=event_links[event])  # submission == post

    # Get all comments from submission (ca 8 mins for 2.5k comments)
    print("|____ Get all comments")
    while True:
        try:
            submission.comments.replace_more(limit=None)
        except Exception:
            print("     |____ Got code 429, retry in 4 seconds")
            time.sleep(4)
            continue
        break

    # Filter necessary attributes
    print("|____ Filter attributes")
    # Attributes of Comment object: https://praw.readthedocs.io/en/latest/code_overview/models/comment.html
    # We want: author, body_html, created_utc, id, parent_id, score
    comments = []
    for comment in submission.comments.list():
        comments.append(
            [
                comment.id,
                comment.parent_id,
                "deleted" if comment.author is None else comment.author.name,
                comment.body.replace("\n", " "),
                comment.score,
                comment.created_utc,
            ]
        )

    # Save comments as DataFrame to csv
    print("|____ Save to file")
    df = pd.DataFrame(
        comments,
        columns=["Comment ID", "Parent ID", "Author", "Body", "Score", "Created (UTC)"],
    )
    df.to_csv(event + "_Data.csv")
# %%
