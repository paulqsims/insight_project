"""
Module for scraping skincare product data from the web. 
"""

#import requests

#URL = 'https://www.paulaschoice.com/ingredient-dictionary'
#page = requests.get(URL)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd

# access chrome driver
driver = webdriver.Chrome('~/Paul/Downloads/chromedriver')