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
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

if os.name == "posix":
    plt.rc("font", family="AppleGothic")

else:
    plt.rc("font", family="Malgun Gothic")


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
for k in tqdm(range(len(movie_name)), mininterval=1, desc="progress_analysis"):
    df = pd.read_csv(movie_name[k] + "_review.csv")

    # 모든 리뷰를 하나의 문자열로 합침
    review_text = df["review"].str.cat(sep=" ")

    # review_text를 text_to_word_sequence()를 이용해 단어 단위로 분리
    tokens = text_to_word_sequence(review_text)
    print("단어 개수:", len(tokens))

    # 불용어 제거
    stop_words = stopwords.words("english")
    tokens = [w for w in tokens if not w in stop_words]

    # 영어가 아닌 단어 제거
    tokens = [word for word in tokens if word.isalpha()]

    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # tokens에 중복된 단어 제거
    tokens = list(set(tokens))

    # glove.6B.100d.txt 파일을 읽어옴
    embeddings_index = {}
    f = open("glove.6B.100d.txt", encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    # 단어 벡터를 저장할 배열 생성
    embedding_matrix = np.zeros((len(tokens), 100))

    # 단어 벡터를 저장
    for i, word in tqdm(enumerate(tokens), desc="progress_embedding"):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 코사인 유사도를 구할 단어 벡터를 저장할 배열 생성
    word_vectors = np.zeros((len(tokens), 100))

    # 코사인 유사도를 구할 단어 벡터를 저장
    for i, word in tqdm(enumerate(tokens), desc="progress_word_vectors"):
        word_vectors[i] = embedding_matrix[i]

    # 코사인 유사도를 구함
    similarity = cosine_similarity(word_vectors)
    print(similarity)

    # 코사인 유사도를 이용해 그래프 생성
    G = nx.from_numpy_array(similarity)
    scores = nx.pagerank(G)

    # 그래프에서 가장 중요한 단어 10개를 출력
    ranked_words = sorted(
        ((scores[i], s) for i, s in tqdm(enumerate(tokens), desc="ranked_words")),
        reverse=True,
    )
    for word in ranked_words[:10]:
        print(word)

    # 저장
    df = pd.DataFrame(ranked_words)
    df.to_csv(movie_name[k] + "_word.csv", index=False)
