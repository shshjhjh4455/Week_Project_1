import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
import jpype
from konlpy.tag import Okt


def rating_transfer(rating):
    if rating > 5:
        rating = 1
    elif rating <= 5:
        rating = 0
    pn = rating
    return pn


review_list = []
rating_list = []
base_url = "https://www.imdb.com/"
key = ""


# 수집하고 싶은 영화의 user riviews 페이지 url 붙여넣기
url = "https://www.imdb.com/title/tt6751668/reviews/?ref_=nv_sr_srsg_0"

# 수집하고 싶은 영화 리뷰 수 지정
MAX_CNT = 30
cnt = 0

print("url = ", url)
res = requests.get(url)
res.encoding = "utf-8"
soup = BeautifulSoup(res.text, "lxml")

for item in soup.select(".lister-list"):
    rating = item.select("span.rating-other-user-rating > span")
    if len(rating) == 2:
        rating = rating[0].text
    else:
        rating = ""
    review = item.select(".text")[0].text

load_more = soup.select(".load-more-data")
flag = True
if len(load_more):
    ajaxurl = load_more[0]["data-ajaxurl"]
    base_url = base_url + ajaxurl + "?ref_=undefined&paginationKey="
    key = load_more[0]["data-key"]
else:
    flag = False

while flag:
    url = base_url + key
    print("url = ", url)
    res = requests.get(url)
    res.encoding = "utf-8"
    soup = BeautifulSoup(res.text, "lxml")
    for item in soup.select(".lister-item-content"):
        rating = item.select("span.rating-other-user-rating > span")
        if len(rating) == 2:
            rating = rating[0].text
            review = item.select(".text")[0].text
            pn = rating_transfer(int(rating))
            review_list.append(review)
            rating_list.append(pn)
            cnt = cnt + 1
        else:
            rating = ""
        review = item.select(".text")[0].text

        if cnt >= MAX_CNT:
            break
    if cnt >= MAX_CNT:
        break
    load_more = soup.select(".load-more-data")
    if len(load_more):
        key = load_more[0]["data-key"]
    else:
        flag = False

df = pd.DataFrame(columns=["review", "label"])
df["review"] = review_list
df["label"] = rating_list
df.to_csv("IMDB_reviews.csv")

df.info()
# 코멘트가 없는 리뷰 데이터(NaN) 제거
df_reviews = df.dropna()
# 중복 리뷰 제거
df_reviews = df_reviews.drop_duplicates(["review"])

# "title" 행 생성
# df_reviews["title"] = "IMDB" # 영화 제목 입력

# 긍정, 부정 리뷰 수
print(df_reviews.groupby("label").size().reset_index(name="count"))

# 긍정, 부정 리뷰 활용하기 위한 변수 선언
pos_reviews = df_reviews[df_reviews["label"] == 1]
neg_reviews = df_reviews[df_reviews["label"] == 0]


# 리뷰 영어 이외의 문자 제거
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return text



# 긍정 리뷰 영어 이외의 문자 제거
pos_reviews["review"] = pos_reviews["review"].map(lambda x: clean_text(x))
# 부정 리뷰 영어 이외의 문자 제거
neg_reviews["review"] = neg_reviews["review"].map(lambda x: clean_text(x))


# 형태소 분석
def tokenize(doc):
    okt = Okt()
    return [" ".join(okt.morphs(s)) for s in doc]


# 긍정 리뷰 형태소 분석
pos_reviews["review"] = tokenize(pos_reviews["review"])
# 부정 리뷰 형태소 분석
neg_reviews["review"] = tokenize(neg_reviews["review"])

# 긍정 리뷰 형태소 분석 결과 확인
print(pos_reviews["review"].head())
# 부정 리뷰 형태소 분석 결과 확인
print(neg_reviews["review"].head())

