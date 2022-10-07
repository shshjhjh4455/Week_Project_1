# https://medium.com/@whj2013123218/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-twitter-%ED%81%AC%EB%A1%A4%EB%A7%81-576f7b098daf

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
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

# 크롤링할 트윗 날짜 설정
startdate = dt.date(year=2019, month=5, day=1)
untildate = dt.date(year=2019, month=5, day=2)
enddate = dt.date(year=2022, month=10, day=4)

totalfreq = []
while not enddate == startdate:
    url = (
        "https://twitter.com/search?q=parasite%20since%3A"
        + str(startdate)
        + "%20until%3A"
        + str(untildate)
        + "&amp;amp;amp;amp;amp;amp;lang=eg"
    )
    browser.get(url)
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")

    lastHeight = browser.execute_script("return document.body.scrollHeight")
while True:
    dailyfreq = {"Date": startdate}
    #     i=0 i는 페이지수
    wordfreq = 0
    tweets = soup.find_all("p", {"class": "TweetTextSize"})
    wordfreq += len(tweets)

    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

    newHeight = browser.execute_script("return document.body.scrollHeight")
    print(newHeight)
    if newHeight != lastHeight:
        html = browser.page_source
        soup = BeautifulSoup(html, "html.parser")
        tweets = soup.find_all("p", {"class": "TweetTextSize"})
        wordfreq = len(tweets)
    else:
        dailyfreq["Frequency"] = wordfreq
        wordfreq = 0
        totalfreq.append(dailyfreq)
        startdate = untildate
        untildate += dt.timedelta(days=1)
        dailyfreq = {}
        break
    #         i+=1
    lastHeight = newHeight
