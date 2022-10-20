"""
참고
https://wikidocs.net/91051
"""

import numpy as np
import nltk
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    "미나리",
    "오징어 게임",
    "이상한 변호사 우영우",
    "Busanhaeng",
    "Gaetmaeul Chachacha",
]
for i in tqdm(range(len(movie_name)), mininterval=1, desc="progress_analysis"):
    df = pd.read_csv(movie_name[i] + "_review.csv")

    # 모든 리뷰를 하나의 문자열로 합침
    review_text = df["review"].str.cat(sep=" ")

    # review_text를 text_to_word_sequence()를 이용해 단어 단위로 분리
    tokens = text_to_word_sequence(review_text)
    print("단어 개수:", len(tokens))

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

    # 문장에 대해서 calculate_sentence_vector()를 적용
    sentences = sent_tokenize(review_text)
    sentence_vectors = [
        calculate_sentence_vector(text_to_word_sequence(sentence))
        for sentence in sentences
    ]

    # 문장 벡터 간의 코사인 유사도 행렬을 구한다.
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 코사인 유사도 행렬을 이용해 그래프를 생성한다.
    graph = nx.from_numpy_array(similarity_matrix)

    # 그래프 출력(숫자 라벨 지우기), 그래프 저장
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(graph, with_labels=False, node_size=10)
    plt.savefig(movie_name[i] + "_graph.png")

    # 그래프에서 문장의 중요도를 계산한다.
    scores = nx.pagerank(graph)

    # 페이저랭크 알고리즘으로 구한 점수를 기준으로 문장을 정렬
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in tqdm(enumerate(sentences), desc="sorted by score")),
        reverse=True,
    )

    # 상위 3개의 문장을 출력
    for i in range(3):
        print(ranked_sentences[i][1])
        print()

    # csv 파일로 저장
    df = pd.DataFrame(ranked_sentences)
    df.to_csv(movie_name[i] + "_analysis.csv", index=False, encoding="utf-8-sig")
    print(movie_name[i] + "_analysis.csv 파일 저장 완료")
