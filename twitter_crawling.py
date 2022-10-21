# https://developer.twitter.com/en/portal/projects/1582545735115243521/apps/25776793/keys

import tweepy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from collections import Counter
from PIL import Image
import datetime
import time
import pytz
from tqdm import tqdm
from pyspark.sql.functions import unix_timestamp, from_unixtime

# 트위터 개발자 계정에서 발급받은 키와 토큰을 입력합니다.
consumer_key = "Zvo63VWQx8wGsYuyjIK4qCazF"
consumer_secret = "dpqekquWb9hUUGkiPmGZKaypdSaZDDtpdAwMLn3eukeMC5z6n9"
access_token = "1537768190616686592-q6WpKSQv51VAtUwuNPjLb7xlLIaGqi"
access_token_secret = "5qnkynxGIBiiW6JxeYbZhEhRbNkWpQ78O7fixiXcNHVuj"


# 트위터 API 인증을 합니다.
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# 트위터 검색 API(api.search_tweets)를 이용해 트윗을 검색합니다.
# 검색어 : "parasite"
# 검색 기간 : 2021-01-01 ~ 2021-01-31

# 검색어를 입력합니다.
keyword = "movie"

# API.search_tweets(q, geocode, lang, locale, result_type, count, until, since_id, max_id, include_entities)
# q : 검색어 = keyword
# lang : 언어 = ko
# result_type : 최신순, 인기순 = recent
# count : 검색 결과 수 = 100
# until : 검색 기간 = 2021-01-31

# 검색 결과를 저장할 tweets 배열을 생성합니다.
tweets = []

# 검색 결과를 tweets 배열에 저장합니다.
for tweet in tqdm(tweepy.Cursor(api.search_tweets, q=keyword, lang="en", result_type="recent", count=100, until="2022-10-21").items()):
    tweets.append(tweet)

# 트윗의 개수를 출력합니다.
print("트윗 개수 : ", len(tweets))
