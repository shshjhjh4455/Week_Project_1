# 트위터 트윗 크롤링하여 그래프로 시각화하기

import tweepy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from collections import Counter
from PIL import Image
from datetime import datetime

# 트위터 개발자 계정에서 발급받은 키와 토큰을 입력합니다.
consumer_key = "qWTxC5XopETX9TLtR6N8kPXIM"
consumer_secret = "UeNkdgXggLBYF2lOm2rK4rbH0htlUf04QJW2OjrtpAmaOW02Mr"
access_token = "1537768190616686592-AVeMPjPU6nINNqPESzoz32GIKacdsB"
access_token_secret = "L4YOH7dhXUw2uVg3hbQRVG1OvdmMZSGhV5bHInD8vTGuy"


# 트위터 API 인증을 합니다.
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweets = []
tmpTweets = api.search_tweets("hashtags and filteration")

# startDate(2019-01-01)
startDate = datetime(2019, 1, 1, 0, 0, 0)
# endDate(2019-12-31)
endDate = datetime(2019, 12, 31, 0, 0, 0)


for tweet in tmpTweets:
    if tweet.created_at < endDate and tweet.created_at > startDate:
        tweets.append(tweet)


# 트윗 리스트를 데이터프레임으로 만듭니다.
df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=["Tweets"])

# 트윗 수를 카운트하여 그래프로 그립니다.
