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

    # 단어가 숫자인 경우 제거
    tokens = [word for word in tokens if word.isalpha()]

    # tokens에 중복된 단어 제거
    tokens = list(set(tokens))

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

    # tokens 리스트를 이용해 단어 벡터 표현을 만든다.
    word_vector_dict = dict()
    for word in tqdm(tokens, desc="word_vector"):
        if word in glove_dict:
            word_vector_dict[word] = glove_dict[word]
        else:
            word_vector_dict[word] = zero_vector
    
    # word_vector_dict를 이용해 각 단어의 중요도를 계산한다.
    word_vector_df = pd.DataFrame(word_vector_dict).T
    word_vector_df["word"] = word_vector_df.index
    word_vector_df.columns = list(range(100)) + ["word"]
    word_vector_df = word_vector_df.reset_index(drop=True)

    # 각 단어의 중요도를 계산한다.
    word_vector_df["importance"] = word_vector_df.iloc[:, 0:100].sum(axis=1)

    # 중요도가 높은 단어 순으로 정렬한다.
    word_vector_df = word_vector_df.sort_values(by="importance", ascending=False)

    # 중요도가 높은 단어 10개를 출력한다.
    print(word_vector_df.head(10))
    
    # csv 파일로 저장
    word_vector_df.to_csv(movie_name[k] + "_word_vector.csv", index=False)
