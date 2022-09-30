from collections import Counter
import requests
import nltk
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# def rating_transfer(rating):
#     if rating > 5:
#         rating = 1
#     elif rating <= 5:
#         rating = 0
#     pn = rating
#     return pn


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
            pn = int(rating)
            rating_list.append(pn)
            review_list.append(review)
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

df = pd.DataFrame(columns=["label","review" ])
df["review"] = review_list
df["label"] = rating_list
df.to_csv("IMDB_reviews.csv")

df.info()

## 문장 토큰화
#review 컬럼을 반복하여 sent_tokenize()를 이용해 문장 단위로 분리
sentences = []
for s in df['review']:
    sentences.extend(sent_tokenize(s))
print("분리된 문장 개수:", len(sentences))

'''
#문장 단위로 분리된 데이터를 확인
for i in range(len(sentences)):
    print('{}번 째 문장: {}\n'.format(i+1, sentences[i]))
'''

##단어 토큰화
#문장 단위로 분리된 데이터를 반복하여 text_to_word_sequence() 를 이용해 단어 단위로 분리
words = []
for sentence in sentences:
    temp_X = text_to_word_sequence(sentence)
    words.extend(temp_X)

'''
#단어 단위로 분리된 데이터를 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))
'''

##불용어 제거
#불용어 목록을 불러온 후, 단어 단위로 분리된 데이터를 반복하여 불용어 제거
stop_words = nltk.corpus.stopwords.words('english')
words = [word for word in words if word not in stop_words]

# words의 단어 길이가 1이하인 단어 제거
words = [word for word in words if len(word) > 1]

# words의 단어가 숫자인 단어 제거
words = [word for word in words if not word.isnumeric()]

# words의 단어가 영어가 아닌 단어 제거
words = [word for word in words if word.isalpha()]

# words의 단어를 소문자로 변환
words = [word.lower() for word in words]

# words의 단어의 품사가 명사, 형용사, 부사, 동사인 단어만 추출
tagged = nltk.pos_tag(words)
words = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]

##불용어 제거 후 단어 개수 확인
print("불용어 제거 후 단어 개수:", len(words))

'''
##불용어 제거 후 단어 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))
'''

##표제어 추출
#단어 단위로 분리된 데이터를 반복하여 표제어 추출
lemmatizer = nltk.stem.WordNetLemmatizer()
words = [lemmatizer.lemmatize(w) for w in words]

##표제어 추출 후 단어 개수 확인
print("표제어 추출 후 단어 개수:", len(words))


##표제어 추출 후 단어 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))

##단어 빈도수 확인
#단어 단위로 분리된 데이터를 반복하여 단어 빈도수 확인
word_count = Counter(words)
print("단어 빈도수:", word_count)



#######textrank 사용해서 수정!

