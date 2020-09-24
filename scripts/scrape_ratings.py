"""
Module for scraping skincare product ratings data from the web. 
"""

# import modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 
import pandas as pd
import rootpath
from bs4 import BeautifulSoup
import math

# Set root path for project
path = rootpath.detect()

# access chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

# Generate urls for different ratings
ratings_webpages = []
for page in range(1,6):
    ratings_webpages.append(f"https://www.beautypedia.com/skin-care/?size=96&rating[0]={page}")

#### Access website for ratings of 1 ####
driver.get(ratings_webpages[0])

# Extract html link references 
time.sleep(5)
item_list = driver.find_elements_by_xpath("//div[@class='review-result']//div[@class='review-details']//div[@class='review-col col-2']//a[starts-with(@href, 'https://www.beautypedia.com/products/')]")

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
item_names = []
item_ingredients = []
item_rating = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    item_name_element = driver.find_elements_by_xpath("//h1[@class='product-name']")[0].text
    item_names.append(item_name_element)
    try:
        ingredient_element = driver.find_element_by_css_selector("div.ingredients").get_attribute("textContent")
    except NoSuchElementException: 
        ingredient_element = math.nan
    item_ingredients.append(ingredient_element)
    # CHANGE ratings-stars number for each rating score!!!!
    rating_element = driver.find_element_by_xpath("//span[@class='product-rating rating stars-1']").get_attribute("outerHTML")
    item_rating.append(rating_element)

# Combine lists into dataframe and export as CSV for pandas processing
df = pd.DataFrame(columns=['Product','Ingredients','Rating']) # creates master dataframe 

data_tuples = list(zip(item_names[1:],item_ingredients[1:],item_rating[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Product', 'Ingredients','Rating']) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

df.to_csv(f"{path}/data/rating1_scrape.csv")

# df.to_csv(f"{path}/data/rating{page}_scrape.csv")

#### Access website for ratings of 2 ####

driver = webdriver.Chrome(f"{path}/chromedriver")

driver.get(ratings_webpages[1])

# Extract html link references 
item_list = driver.find_elements_by_xpath("//div[@class='review-result']//div[@class='review-details']//div[@class='review-col col-2']//a[starts-with(@href, 'https://www.beautypedia.com/products/')]")

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
item_names = []
item_ingredients = []
item_rating = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    item_name_element = driver.find_elements_by_xpath("//h1[@class='product-name']")[0].text
    item_names.append(item_name_element)
    try:
        ingredient_element = driver.find_element_by_css_selector("div.ingredients").get_attribute("textContent")
    except NoSuchElementException: 
        ingredient_element = math.nan
    item_ingredients.append(ingredient_element)
    # CHANGE ratings-stars number for each rating score!!!!
    rating_element = driver.find_element_by_xpath("//span[@class='product-rating rating stars-2']").get_attribute("outerHTML")
    item_rating.append(rating_element)

# Combine lists into dataframe and export as CSV for pandas processing
df = pd.DataFrame(columns=['Product','Ingredients','Rating']) # creates master dataframe 

data_tuples = list(zip(item_names[1:],item_ingredients[1:],item_rating[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Product', 'Ingredients','Rating']) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

df.to_csv(f"{path}/data/rating2_scrape.csv")

#### Access website for ratings of 3 ####

driver = webdriver.Chrome(f"{path}/chromedriver")

driver.get(ratings_webpages[2])

# Extract html link references 
time.sleep(5)
item_list = driver.find_elements_by_xpath("//div[@class='review-result']//div[@class='review-details']//div[@class='review-col col-2']//a[starts-with(@href, 'https://www.beautypedia.com/products/')]")

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
item_names = []
item_ingredients = []
item_rating = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    item_name_element = driver.find_elements_by_xpath("//h1[@class='product-name']")[0].text
    item_names.append(item_name_element)
    try:
        ingredient_element = driver.find_element_by_css_selector("div.ingredients").get_attribute("textContent")
    except NoSuchElementException: 
        ingredient_element = math.nan
    item_ingredients.append(ingredient_element)
    # CHANGE ratings-stars number for each rating score!!!!
    rating_element = driver.find_element_by_xpath("//span[@class='product-rating rating stars-3']").get_attribute("outerHTML")
    item_rating.append(rating_element)
    
# Combine lists into dataframe and export as CSV for pandas processing
df = pd.DataFrame(columns=['Product','Ingredients','Rating']) # creates master dataframe 

data_tuples = list(zip(item_names[1:],item_ingredients[1:],item_rating[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Product', 'Ingredients','Rating']) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

df.to_csv(f"{path}/data/rating3_scrape.csv")

#### Access website for ratings of 4 ####

driver = webdriver.Chrome(f"{path}/chromedriver")
driver.get(ratings_webpages[3])

# Extract html link references 
time.sleep(5)
item_list = driver.find_elements_by_xpath("//div[@class='review-result']//div[@class='review-details']//div[@class='review-col col-2']//a[starts-with(@href, 'https://www.beautypedia.com/products/')]")

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
item_names = []
item_ingredients = []
item_rating = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    item_name_element = driver.find_elements_by_xpath("//h1[@class='product-name']")[0].text
    item_names.append(item_name_element)
    try:
        ingredient_element = driver.find_element_by_css_selector("div.ingredients").get_attribute("textContent")
    except NoSuchElementException: 
        ingredient_element = math.nan
    item_ingredients.append(ingredient_element)
    # CHANGE ratings-stars number for each rating score!!!!
    rating_element = driver.find_element_by_xpath("//span[@class='product-rating rating stars-4']").get_attribute("outerHTML")
    item_rating.append(rating_element)

# Combine lists into dataframe and export as CSV for pandas processing
df = pd.DataFrame(columns=['Product','Ingredients','Rating']) # creates master dataframe 

data_tuples = list(zip(item_names[1:],item_ingredients[1:],item_rating[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Product', 'Ingredients','Rating']) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

df.to_csv(f"{path}/data/rating4_scrape.csv")

#### Access website for ratings of 5 ####

driver = webdriver.Chrome(f"{path}/chromedriver")
driver.get(ratings_webpages[4])

# Extract html link references 
time.sleep(5)
item_list = driver.find_elements_by_xpath("//div[@class='review-result']//div[@class='review-details']//div[@class='review-col col-2']//a[starts-with(@href, 'https://www.beautypedia.com/products/')]")

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
item_names = []
item_ingredients = []
item_rating = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    item_name_element = driver.find_elements_by_xpath("//h1[@class='product-name']")[0].text
    item_names.append(item_name_element)
    try:
        ingredient_element = driver.find_element_by_css_selector("div.ingredients").get_attribute("textContent")
    except NoSuchElementException: 
        ingredient_element = math.nan
    item_ingredients.append(ingredient_element)
    # CHANGE ratings-stars number for each rating score!!!!
    rating_element = driver.find_element_by_xpath("//span[@class='product-rating rating stars-5']").get_attribute("outerHTML")
    item_rating.append(rating_element)

# Combine lists into dataframe and export as CSV for pandas processing
df = pd.DataFrame(columns=['Product','Ingredients','Rating']) # creates master dataframe 

data_tuples = list(zip(item_names[1:],item_ingredients[1:],item_rating[1:])) # list of each ingredient with ratings and categories paired
temp_df = pd.DataFrame(data_tuples, columns=['Product', 'Ingredients','Rating']) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

df.to_csv(f"{path}/data/rating5_scrape.csv")









