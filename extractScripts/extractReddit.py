import praw
import time
import pandas as pd
import os
from datetime import datetime, timedelta

exportPath = "C:/Projects/pyStock/Data/RedditPosts.csv"

df_RedditPosts = pd.read_csv(exportPath)

filterDate = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days = 14)

# Reddit API credentials
reddit = praw.Reddit(
    client_id="vTujgjL24YYX-R80GyoKhg",
    client_secret="mDGxUHWOxPt-pKNUVZoLcNPzNaBu6w",
    password="Cooper123",
    user_agent="test",
    username="xXHomelanderXx",
)
# Subreddits to scrape
subreddit_names = [
                   "Daytrading",
                   "investing",
                   "options",
                   "pennystocks",
                   "RobinHood",
                   "StockMarket",
                   "stocks",
                   "wallstreetbets",
                   "wallstreetbetsnew"
                   ]


# extract and append new posts for selected subreddits
for sub in subreddit_names:
    for submission in reddit.subreddit(sub).new(limit=None):

        post_date = submission.created_utc  # UTC timestamp of the post creation date

        if post_date < filterDate.timestamp():
            print("filter date reached for " + sub)
            break

        tmp = {
            "post_title": submission.title,
            "subreddit" : sub,
            "date"      : post_date
        }

        df_RedditPosts.loc[len(df_RedditPosts)] = tmp

# export updated table

df_export = df_RedditPosts.drop_duplicates().sort_values(by=["subreddit", "date"], ascending= [True,True])

df_export.to_csv(exportPath, index=False)

