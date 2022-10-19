"""
참고
https://wikidocs.net/91051
"""

import numpy as np
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile
from collections import Counter
import requests
import nltk
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
zf = zipfile.ZipFile('glove.6B.zip')
zf.extractall()
zf.close()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

def rating_transfer(rating):
    if rating > 8:
        rating = 1
    elif rating <= 3:
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

for item in soup.select(".lister-list"):  # 리뷰가 있는곳
    rating = item.select("span.rating-other-user-rating > span")  # 평점이 있는곳
    if len(rating) == 2:  # 평점이 있는 리뷰만
        rating = rating[0].text  # 평점
    else:  # 평점이 없는 리뷰는 제외
        rating = ""
    review = item.select(".text")[0].text  # 리뷰글



load_more = soup.select(
    ".load-more-data"
)  # www.imdb.com/title/tt6751668/reviews/?ref_=nv_sr_srsg_0 파일에 load-more-data 클래스
flag = True
if len(load_more):  # load-more-data 클래스가 있으면
    ajaxurl = load_more[0]["data-ajaxurl"]  # data-ajaxurl 속성값을 가져온다.
    base_url = (
        base_url + ajaxurl + "?ref_=undefined&paginationKey="
    )  # base_url에 붙여준다. ?ref_=undefined&paginationKey= 속성값은 베이스 url,ajaxurl, 키값만 사용하고 나머지는 빼준다.
    key = load_more[0]["data-key"]  # data-key 속성값을 가져온다.
else:  # load-more-data 클래스가 없으면
    flag = False  # load-more-data 클래스가 없으면 flag를 False로 바꾼다.

while flag:  # flag가 True이면
    url = base_url + key  # url을 만든다.
    print("url = ", url)  # url을 출력한다.
    res = requests.get(url)  # url을 요청한다.
    res.encoding = "utf-8"  # 인코딩을 utf-8로 한다.
    soup = BeautifulSoup(res.text, "lxml")  # soup을 만든다.
    for item in soup.select(".lister-item-content"):  # 리뷰가 있는곳
        rating = item.select("span.rating-other-user-rating > span")  # 평점이 있는곳
        if len(rating) == 2:  # 평점이 있는 리뷰만
            rating = rating[0].text  # 평점
            review = item.select(".text")[0].text  # 리뷰글
            pn = int(rating)  # 평점을 정수로 바꾼다.
            rating_list.append(pn)  # 평점을 rating_list에 추가한다.
            review_list.append(review)  # 리뷰를 review_list에 추가한다.
            cnt = cnt + 1
        else:  # 평점이 없는 리뷰는 제외
            rating = ""
        review = item.select(".text")[0].text

        if cnt >= MAX_CNT:  # cnt가 MAX_CNT보다 크거나 같으면
            break
    if cnt >= MAX_CNT:  # cnt가 MAX_CNT보다 크거나 같으면
        break
    load_more = soup.select(".load-more-data")  # load-more-data 클래스가 있는곳
    if len(load_more):  # load-more-data 클래스가 있으면
        key = load_more[0]["data-key"]  # data-key 속성값을 가져온다.
    else:
        flag = False

df = pd.DataFrame(columns=["label", "review"])
df["review"] = review_list
df["label"] = rating_list
df.to_csv("IMDB_reviews.csv")

df.info()

## 문장 토큰화
# review 컬럼을 반복하여 sent_tokenize()를 이용해 문장 단위로 분리
sentences = []
for s in df["review"]:
    sentences.extend(sent_tokenize(s))
print("분리된 문장 개수:", len(sentences))


"""
#문장 단위로 분리된 데이터를 확인
for i in range(len(sentences)):
    print('{}번 째 문장: {}\n'.format(i+1, sentences[i]))
"""


##단어 토큰화
# 문장 단위로 분리된 데이터를 'sentences'열에 대해서 text_to_word_sequence()를 적용한 'tokenized_sentences' 열을 새로 만듭니다.
df["tokenized_sentences"] = df["review"].apply(text_to_word_sequence)

# 사전 훈련된 GloVe
glove_dict = dict()
f = open("glove.6B.100d.txt", encoding="utf8")  # 100차원의 GloVe 벡터를 사용

for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(
        word_vector[1:], dtype="float32"
    )  # 100개의 값을 가지는 array로 변환
    glove_dict[word] = word_vector_arr
f.close()

#  GloVe 벡터의 차원은 100. 100차원의 영벡터를 만든다.
embedding_dim = 100
zero_vector = np.zeros(embedding_dim)

# 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
def calculate_sentence_vector(sentence):
    if len(sentence) != 0:
        return sum([glove_dict.get(word, zero_vector) for word in sentence]) / len(
            sentence
        )
    else:
        return zero_vector


# 각 문장에 대해서 문장 벡터를 반환
def sentences_to_vectors(sentences):
    return [calculate_sentence_vector(sentence) for sentence in sentences]


# 모든 문장에 대해서 문장 벡터를 만든다.
df["SentenceEmbedding"] = df["tokenized_sentences"].apply(sentences_to_vectors)

# 문장 벡터들 간의 코사인 유사도를 구한 유사도 행렬을 만든다.
def similarity_matrix(sentence_embedding):
    sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
    for i in range(len(sentence_embedding)):
        for j in range(len(sentence_embedding)):
            sim_mat[i][j] = cosine_similarity(
                sentence_embedding[i].reshape(1, embedding_dim),
                sentence_embedding[j].reshape(1, embedding_dim),
            )[0, 0]
    return sim_mat


# 이 결과를 저장한 'SimMatrix'열을 만든다.
df["SimMatrix"] = df["SentenceEmbedding"].apply(similarity_matrix)
print(df["SimMatrix"])

""" 
# 유사도 행렬로부터 그래프를 그린다. #시간소요도 높음
def draw_graphs(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, with_labels=True, font_weight="bold")
    nx.draw_networkx_edge_labels(nx_graph, pos, font_color="red")
    plt.show()


draw_graphs(df["SimMatrix"][1])
"""

# 페이지랭크 알고리즘의 입력으로 사용하여 각 문장의 점수를 구한다.
def calculate_score(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores


df["score"] = df["SimMatrix"].apply(calculate_score)
print(df[["SimMatrix", "score"]])

# 이 점수가 가장 높은 문장들을 상위 n개 선택하여 이 문서의 요약문으로 삼을 것이다. 점수가 가장 높은 상위 3개의 문장을 선택하는 함수를 만든다. 점수에 따라서 정렬 후에 상위 3개 문장만을 반환
def ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [sentence for score, sentence in top_scores[:n]]
    return " ".join(top_n_sentences)


# 'ranked_sentences' 함수를 적용하여 'summary'열을 만든다.
df["summary"] = df.apply(
    lambda x: ranked_sentences(x["tokenized_sentences"], x["score"]), axis=1
)
print(df[["tokenized_sentences", "score", "summary"]])


for i in range(0, len(df)):
    print(i + 1, "번 문서")
    print("요약문 : ", df["summary"][i])
    print("")

# 요약문을 .xml 파일로 저장
import xml.etree.ElementTree as ET

root = ET.Element("root")
for i in range(0, len(df)):
    doc = ET.SubElement(root, "doc")
    doc.text = df["summary"][i]

tree = ET.ElementTree(root)
tree.write("summary.xml", encoding="utf-8", xml_declaration=True)


'''


"""
#단어 단위로 분리된 데이터를 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))
"""

##불용어 제거
# 불용어 목록을 불러온 후, 단어 단위로 분리된 데이터를 반복하여 불용어 제거
stop_words = nltk.corpus.stopwords.words("english")
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
words = [
    word
    for word, tag in tagged
    if tag
    in [
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
        "RB",
        "RBR",
        "RBS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
    ]
]

##불용어 제거 후 단어 개수 확인
print("불용어 제거 후 단어 개수:", len(words))

"""
##불용어 제거 후 단어 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))
"""

##표제어 추출
# 단어 단위로 분리된 데이터를 반복하여 표제어 추출
lemmatizer = nltk.stem.WordNetLemmatizer()
words = [lemmatizer.lemmatize(w) for w in words]

##표제어 추출 후 단어 개수 확인
print("표제어 추출 후 단어 개수:", len(words))

"""
##표제어 추출 후 단어 확인
for i in range(len(words)):
    print('{}번 째 단어: {}\n'.format(i+1, words[i]))
"""

##단어 빈도수 확인
# 단어 단위로 분리된 데이터를 반복하여 단어 빈도수 확인
word_count = Counter(words)
print("단어 빈도수:", word_count)


#######textrank 사용해서 수정!

'''
