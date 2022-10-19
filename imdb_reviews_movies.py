from urllib.request import urlretrieve, urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import time
from urllib.parse import urljoin
import pandas as pd
import requests
import os

# Mac OS의 경우와 그 외 OS의 경우로 나누어 설정

if os.name == "posix":
    driver = FirefoxBinary("/Applications/Firefox.app/Contents/MacOS/firefox")
    browser = webdriver.Firefox(firefox_binary=driver)

else:
    driver = FirefoxBinary("C:/Program Files/Mozilla Firefox/firefox.exe")
    browser = webdriver.Firefox(firefox_binary=driver)

# 여러 영화 수집
movie_list = [
    "tt6751668",
    "tt10919420",
    "tt10633456",
    "tt5700672",
    "tt20869502",
    "tt14518756",
]
review_list = []
rating_list = []

# movie_list 사용하여 리뷰 크롤링 시작, 리뷰 평점 높은 순서대로 크롤링
for movie in movie_list:
    url = (
        "https://www.imdb.com/title/"
        + movie
        + "/reviews?sort=userRating&dir=desc&ratingFilter=0"
    )
    print("url = ", url)
    res = requests.get(url)
    res.encoding = "utf-8"
    soup = BeautifulSoup(res.text, "lxml")

    """기본적 방법으로 크롤링"""
    need_reviews_cnt = 30
    review_button_cnt = 3
    reviews = []
    review_data = []

    # button 클릭하여 다음 리뷰 불러오기
    for i in range(review_button_cnt):
        try:
            browser.get(url)
            time.sleep(1)
            button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
            button.click()
            time.sleep(1)
        except:
            print("더 이상 리뷰가 없습니다.")
            break
    #  영화 제목(#main > section > div.subpage_title_block > div > div > h3 > a) 수집
    movie_title = soup.select_one(
        "#main > section > div.subpage_title_block > div > div > h3 > a"
    ).text
    print("movie_title = ", movie_title)

    # 리뷰 수집
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
    reviews = soup.select("div.text.show-more__control")

    # 리뷰 수집
    for review in reviews:
        review_data.append(review.text)

    # 영화별로 데이터 프레임 생성 후 저장
    df = pd.DataFrame({"review": review_data})
    df.to_csv("Week_Project_1/" + movie_title + "_review.csv")

    # 리뷰 수집 완료
    print("수집 완료")

