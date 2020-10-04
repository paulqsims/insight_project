'''
Script for scraping ulta products
'''

# import modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 
import pandas as pd
import rootpath
import math

# Set root path for project
path = rootpath.detect()

# access chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

#### Scrape Cleansers ####

# ---------------------
# Category: Face washes
# Skintype: Oily
# Concern: 
# ---------------------

driver.get('https://www.ulta.com/skin-care-cleansers-face-wash?N=27gsZ1z13p3j')

# Don't need since not writing code to loop through pages
# Generate urls for different ratings
# product_links = []
# for page in range(1,6):
#     ratings_webpages.append(f"https://www.beautypedia.com/skin-care/?size=96&rating[0]={page}")

# Page 1 of 3
driver.get('https://www.ulta.com/skin-care-cleansers-face-wash?N=27gsZ1z13p3j')

# Extract all html link references for webpage
# Wait 5 seconds before extracting them
time.sleep(5)
item_list = driver.find_elements_by_xpath("/html/body/div[1]/div[6]/div[2]/div[2]/div[6]/div/div/ul//div[contains(@class, 'productQvContainer')]/a[@href]")
# Note from //div[contains] is the wildcard part that selects each individual product link

# Convert selenium ref info to href links and store them in a vector
product_links = []
for i, link in enumerate(item_list):
    # print(link.get_attribute('href'))
    # Fetch and store the links
    product_links.append(link.get_attribute('href'))

# Create empty lists to store results in 
prod_names = []
prod_ingredients = []
prod_size = []
prod_price = []


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