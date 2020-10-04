'''
Script for scraping ulta products
'''

# import modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
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

# -----------
# Page 1 of 3
# ----------- 
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

# Create empty lists to store results in for each html element
brand_names = []
prod_names = []
prod_sizes = []
prod_prices = []
prod_details = []
prod_ingredientlists = []
prod_ratings = []
prod_respondrecs = []
prod_reviewtotals = []

# Iterate over links to extract text data
for link in product_links:
    time.sleep(5)
    driver.get(link)
    # Brand name
    brand_name = driver.find_elements_by_xpath("//*[@id='js-mobileBody']/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[1]/a[@class]")[0].text
    brand_names.append(brand_name)
    # Product name
    prod_name = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[2]")[0].text
    prod_names.append(prod_name)
    # Product size
    prod_size = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[1]/p[1]")[0].text
    prod_sizes.append(prod_name)
    # Product price
    prod_price = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[3]/span")[0].text
    prod_prices.append(prod_price)
    # Product details
    prod_detail = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[1]/div/div")[0].text
    prod_details.append(prod_detail)
    # Product ingredients
    prod_ingredientlist = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div")[0].text
    prod_ingredientlists.append(prod_ingredientlist)
    # Product average rating
    try:
        prod_rating = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))]").text
    except NoSuchElementException: 
        prod_rating = math.nan
    prod_ratings.append(prod_rating)
    # Product proportion of respondants who would recommend product to friends
    try:
        prod_respondrec = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]").text
    except NoSuchElementException: 
        prod_respondrec = math.nan
    prod_respondrecs.append(prod_respondrec)
    # Product total number of reviews
    try:
        prod_reviewtotal = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]").text
    except NoSuchElementException: 
        prod_reviewtotal = math.nan
    prod_reviewtotals.append(prod_reviewtotal)

# Combine product info lists into dataframe and export as CSV for pandas processing
df = (pd.DataFrame(columns=['brand','product','size', 'price', 'details', 
                            'ingredients', 'ratings', 'perc_respondrec', 'total_reviews'])) # creates master dataframe 

# list of each ingredient with ratings and categories paired
data_tuples = (list(zip(brand_names[1:],prod_names[1:],prod_sizes[1:],
                       prod_prices[1:], prod_details[1:],
                       prod_ingredientlists[1:], prod_ratings[1:],
                       prod_respondrecs[1:], prod_reviewtotals[1:]))) 
# Create dataframe of tuple lists
temp_df = (pd.DataFrame(data_tuples,
                        columns=['brand','product','size', 'price', 'details',
                                 'ingredients', 'ratings', 'perc_respondrec', 'total_reviews'])) # creates dataframe of each tuple in list
df = df.append(temp_df)

driver.close()

# Export to csv
df.to_csv(f"{path}/data/cleansers_face-wash_oilyskin_pg1.csv")

# Test Area ----

driver.get('https://www.ulta.com/face-cleanser?productId=xlsImpprod13491007')

# Brand name
brand_name = driver.find_elements_by_xpath("//*[@id='js-mobileBody']/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[1]/a[@class]")[0].text

# Product name
prod_name = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[2]")[0].text

# Product size
prod_size = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[1]/p[1]")[0].text

# Product price
prod_price = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[3]/span")[0].text

# Product details
prod_detail = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[1]/div/div")[0].text

# Product ingredients
prod_ingredientlist = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div")[0].text

# Product rating
prod_rating = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))]").text

# Prop of respondents who would recommend product
prod_respondrec = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]").text

# Total number of reviews
prod_totalreview = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]").text

driver.get('https://www.ulta.com/pineapple-enzyme-pore-clearing-cleanser?productId=pimprod2018750')

product_ratings = []
try:
    product_rating = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))]").text
except NoSuchElementException: 
    product_rating = math.nan
product_ratings.append(product_rating)





# End test ----
