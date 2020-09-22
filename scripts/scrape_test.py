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

ingredient_names = driver.find_elements_by_xpath('//h2[@class="name ingredient-name"]')

ingredients_list = []
for ingredient in range(len(ingredient_names)):
    ingredients_list.append(ingredient_names[ingredient].text)

ingredient_descript = driver.find_elements_by_xpath('//p[@class="description ingredient-description"]')

ingredient_descript_list = []
for description in range(len(ingredient_descript)):
    ingredient_descript_list.append(ingredient_descript[description].text)

ingredient_rating = driver.find_elements_by_xpath('//td[@class="col-rating ingredient-rating rating-good"]')

ingredient_ratings = []
for rating in range(len(ingredient_rating)):
    ingredient_ratings.append(ingredient_rating[rating].text)

ingredient_cat = driver.find_elements_by_xpath('//div[@class="categories ingredient-categories"]')

ingredient_categories = []
for category in range(len(ingredient_cat)):
    ingredient_categories.append(ingredient_cat[category].text)

df = pd.DataFrame(columns=['Ingredient','Description','Rating','Category']) # creates master dataframe 

data_tuples = list(zip(ingredients_list[1:],ingredient_descript_list[1:],ingredient_ratings[1:],
ingredient_categories[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Ingredient', 'Description','Rating','Category']) # creates dataframe of each tuple in list
df = df.append(temp_df) # appends to master dataframe

driver.close()

df.to_csv(f"{path}/data/test_scrape.csv")