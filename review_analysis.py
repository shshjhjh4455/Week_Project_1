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
from tqdm import tqdm
import time


# urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
# zf = zipfile.ZipFile('glove.6B.zip')
# zf.extractall()
# zf.close()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

# movie_name 배열 생성
movie_name = [
    "미나리_review.csv",
    "오징어 게임_review.csv",
    "이상한 변호사 우영우_review.csv",
    "Busanhaeng_review.csv",
    "Gaetmaeul Chachacha_review.csv",
]
for i in tqdm(range(len(movie_name)), mininterval=1, desc="progress_analysis"):
    df = pd.read_csv(movie_name[i])
    ## 문장 토큰화
    # review 컬럼을 반복하여 sent_tokenize()를 이용해 문장 단위로 분리
    sentences = []
    for s in tqdm(df["review"], desc="sentence_tokenization"):
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

    for line in tqdm(f, desc="glove"):
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
        for i in tqdm(range(len(sentence_embedding)), desc="similarity_matrix"):
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

    # 이 점수가 가장 높은 문장들을 상위 n개 선택하여 이 문서의 요약문으로 삼을 것이다. 점수가 가장 높은 상위 n개의 문장을 선택하는 함수를 만든다. 점수에 따라서 정렬 후에 상위 n개 문장만을 반환
    def ranked_sentences(sentences, scores, n=5):
        top_scores = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
        )
        top_n_sentences = [sentence for score, sentence in top_scores[:n]]
        return " ".join(top_n_sentences)

    # 'ranked_sentences' 함수를 적용하여 'summary'열을 만든다.
    df["summary"] = df.apply(
        lambda x: ranked_sentences(x["tokenized_sentences"], x["score"]), axis=1
    )
    print(df[["tokenized_sentences", "score", "summary"]])

    for k in range(0, len(df)):
        print(k + 1, "번 문서")
        print("요약문 : ", df["summary"][k])
        print("")

    # 요약문을 csv 파일로 저장, 파일명(movie_name[i]_summary.csv)])
    df.to_csv(movie_name[i] + "_summary.csv", index=False, encoding="utf-8-sig")
    print(movie_name[i] + "저장완료")


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
