# firefox 에서 (영화 제목 + 단어리스트)를 검색하여 이미지 1개 얻고 이를 데이터 저장

# 영화 제목과 단어 리스트 이중 배열로 생성(미나리(extreme, immigration, experience, representation, village, prankster, tender, culture, misbehaviour,award,composer,superb),
#  오징어게임(genuine,india,design,mediocre,debt,nameless,blockbuster,greed,dispense,thriller,fearless),
# 이상한변호사우영우(claim,express,genius,heart,politics,sassy,autism,refreshing,opportunity,kdramas,suffers,struggle),)


from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
from tqdm import tqdm
from selenium.webdriver.common.keys import Keys
import datetime as dt
import matplotlib.pyplot as plt
import os

# Mac OS의 경우와 그 외 OS의 경우로 나누어 설정

if os.name == "posix":
    plt.rc("font", family="AppleGothic")
    binary = FirefoxBinary("/Applications/Firefox.app/Contents/MacOS/firefox")
    browser = webdriver.Firefox(firefox_binary=binary)

else:
    plt.rc("font", family="Malgun Gothic")
    binary = FirefoxBinary("C:/Program Files/Mozilla Firefox/firefox.exe")
    browser = webdriver.Firefox(firefox_binary=binary)

# movie_name 배열 생성
movie_name = [
    "미나리",
    "오징어 게임",
    "이상한 변호사 우영우",
    "부산행",
    "갯마을 차차차",
    "기생충",
]

# des_list 배열 생성(영화별 단어 리스트 각 5개)
des_list = [
    [
        "exreme",
        "immigration",
        "experience",
        "representation",
        "village",
        "prankster",
        "tender",
        "culture",
        "misbehaviour",
        "award",
        "composer",
        "superb",
    ],
    [
        "genuine",
        "india",
        "mediocre",
        "design",
        "debt",
        "blockbuster",
        "nameless",
        "greed",
        "dispense",
        "thriller",
        "fearless",
    ],
    [
        "claim",
        "autism",
        "express",
        "genius",
        "sassy",
        "heart",
        "struggle",
        "suffers",
        "refreshing",
        "opportunity",
        "kdramas",
    ],
]

# firefox에서 이미지만 검색하기, movie_name[i] 생성, des_list[i][j]를 검색어로 입력
for i in tqdm(range(len(movie_name)), desc="영화명"):

    for j in tqdm(range(len(des_list[i])), desc="검색어", leave=False):
        browser.get("https://www.google.com/imghp?hl=ko&tab=ri&ogbl")
        browser.find_element_by_name("q").send_keys(
            "영화 " + movie_name[i] + " " + des_list[i][j]
        )
        browser.find_element_by_name("q").send_keys(Keys.ENTER)
        time.sleep(1)

        # 이미지 크롤링
        elem = browser.find_element_by_xpath(
            '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img'
        )
        attr = elem.get_attribute("src")
        # print(movie_name[i] + " 이미지 : " + attr)
        time.sleep(2)

        # 이미지 저장하기
        try:
            resp = requests.get(attr, stream=True).raw
            photo = open(
                "data/{}/{}.jpg".format(movie_name[i], des_list[i][j]),
                "wb",
            )
            photo.write(resp.read())
            photo.close()
        except:
            print("이미지 저장 실패")
            pass

        time.sleep(1)

# 크롤링완료 창 닫기
browser.quit()
print("수집완료")
