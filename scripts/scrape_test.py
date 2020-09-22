"""
Module for scraping skincare product data from the web. 
"""

#import requests

#URL = 'https://www.paulaschoice.com/ingredient-dictionary'
#page = requests.get(URL)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import rootpath
#from src.utils import get_project_root

#root = get_project_root()
path = rootpath.detect()

# access chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

# Access website
driver.get('https://www.paulaschoice.com/ingredient-dictionary')

ingredient_names = driver.find_elements_by_xpath('//p[@class="description ingredient-description"]')

ingredients_list = []
for ingredient in range(len(ingredient_names)):
    ingredients_list.append(ingredient_names[ingredient].text)

ingredient_rating = driver.find_elements_by_xpath('//td[@class="col-rating ingredient-rating rating-good"]')

ingredient_ratings = []
for rating in range(len(ingredient_ratings)):
    ingredient_ratings.append(ingredient_ratings[rating].text)

ingredient_cat = driver.find_elements_by_xpath('//div[@class="categories ingredient-categories"]')

ingredient_categories = []
for category in range(len(ingredient_cat)):
    ingredient_categories.append(ingredient_categories[category].text)

data_tuples = list(zip(players_list[1:],salaries_list[1:])) # list of each players name and salary paired together
temp_df = pd.DataFrame(data_tuples, columns=['Player','Salary']) # creates dataframe of each tuple in list
temp_df['Year'] = yr # adds season beginning year to each dataframe
df = df.append(temp_df) # appends to master dataframe
    
driver.close()