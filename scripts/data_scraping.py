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
from selenium.common.exceptions import TimeoutException
import time 
import pandas as pd
import rootpath
import math

# Set root path for project
path = rootpath.detect()



#### Scrape Cleansers ####

# ---------------------
# Category: Face washes
# Skintype: Oily
# Concern: 
# ---------------------

# -----------
# Page 1 of 3
# ----------- 

# Start up chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

# Fetch product page link, pg 1
driver.get('https://www.ulta.com/skin-care-cleansers-face-wash?N=27gsZ1z13p3j')

# Extract all html link references for webpage
# Wait 5 seconds for page to load before extracting them
time.sleep(5)
item_list = driver.find_elements_by_xpath("/html/body/div[1]/div[6]/div[2]/div[2]/div[6]/div/div/ul//div[contains(@class, 'productQvContainer')]/a[@href]")
# Note from //div[contains] is the wildcard part that selects each individual product link

# Convert selenium refs info to href links and store them in a vector
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

# Iterate over links from webpage of products to extract text data
for link in product_links:
    # wait 5 seconds before going to next link
    time.sleep(5)
    driver.get(link)
    # wait 5 seconds before scraping elements of webpage
    # e.g. allow it to load
    time.sleep(5)
    # Brand name
    brand_name = driver.find_elements_by_xpath("//*[@id='js-mobileBody']/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[1]/a[@class]")[0].text
    brand_names.append(brand_name)
    # Product name
    prod_name = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[2]")[0].text
    prod_names.append(prod_name)
    # Product size
    # If else variant for whether product is one size or has multiple sizes
    if driver.find_elements_by_xpath("//*[contains(@class,'ProductDetail__productVariantOptions')]"):
        prod_size = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[3]/div/div[1]/div[2]/span[@class='Text Text--body-2 Text--left Text--small']")[0].text
    else:
        prod_size = driver.find_elements_by_xpath("//html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[1]/p[1]")[0].text
    # Another option to print all sizes for product variant options
    # In reserve if above if statement does not work
    # driver.find_elements_by_xpath("//*[contains(@class,'ProductDetail__productVariantOptions')]")[0].text
    prod_sizes.append(prod_size)    
    # Product price
    prod_price = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[contains(@class, 'ProductPricingPanel')]")[0].text
    prod_prices.append(prod_price)
    # Product details
    prod_detail = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[1]/div/div")[0].text
    prod_details.append(prod_detail)
    # Product ingredients
    try:
        prod_ingredientlist = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div")[0].get_attribute("innerText")
    except (NoSuchElementException, IndexError): 
        prod_ingredientlist = math.nan
    prod_ingredientlists.append(prod_ingredientlist)
    # Product average rating
    # For this element and below, use webdriverwait to ensure elements have loaded
    # Include try and except for new products that don't have reviews
    WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))] | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[5]/div/div[2]/div[3]/div/section/header/section/div/div[1]/div/div[1]/div/div[2]")))
    try:
        prod_rating = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))] | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[5]/div/div[2]/div[3]/div/section/header/section/div/div[1]/div/div[1]/div/div[2]").text
    except (NoSuchElementException,TimeoutException): 
        prod_rating = math.nan
    prod_ratings.append(prod_rating)
    # Product proportion of respondants who would recommend product to friends
    WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]")))
    try:
        prod_respondrec = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]").text
    except (NoSuchElementException,TimeoutException): 
        prod_respondrec = math.nan
    prod_respondrecs.append(prod_respondrec)
    # Product total number of reviews
    WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]")))
    try:
        prod_reviewtotal = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]").text
    except (NoSuchElementException,TimeoutException): 
        prod_reviewtotal = math.nan
    prod_reviewtotals.append(prod_reviewtotal)

# Create lists for 'use_category', 'use_subcategory', 'skintype', page
use_categories = []
for string in range(len(prod_names)):
    use_categories.append('cleanser')

use_subcategory = []
for string in range(len(prod_names)):
    use_subcategory.append('face wash')

skintype = []
for string in range(len(prod_names)):
    skintype.append('oily')

page = []
for string in range(len(prod_names)):
    page.append(1)

# Combine product info lists into dataframe and export as CSV for pandas processing
df_pg1 = (pd.DataFrame(columns=['use_category', 'use_subcategory', 'skintype',
                            'brand','product','size', 'price', 'details', 
                            'ingredients', 'ratings', 'perc_respondrec', 'total_reviews', 'link', 'page'])) # creates master dataframe 

# list of each ingredient with ratings and categories paired
data_tuples = (list(zip(use_categories[1:],use_subcategory[1:],
                        skintype[1:], brand_names[1:],prod_names[1:],
                        prod_sizes[1:], prod_prices[1:], prod_details[1:],
                        prod_ingredientlists[1:], prod_ratings[1:],
                        prod_respondrecs[1:], prod_reviewtotals[1:],
                        product_links[1:], page[1:]))) 

# Create dataframe of tuple lists
temp_df = (pd.DataFrame(data_tuples,
                        columns=['use_category', 'use_subcategory', 'skintype',
                                 'brand','product','size', 'price', 'details', 
                                 'ingredients', 'ratings', 'perc_respondrec', 'total_reviews', 'link', 'page'])) # creates dataframe of each tuple in list
df_pg1 = df_pg1.append(temp_df)

# Correct Banila Co
df_pg1['size'][11] = '3.3 oz'

driver.close()

# Export to csv
df_pg1.to_csv(f"{path}/data/cleansers_face-wash_oilyskin_pg1.csv")

# -----------
# Page 2 of 3
# ----------- 

# Start up chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

# Fetch product page link, pg 2
driver.get('https://www.ulta.com/skin-care-cleansers-face-wash?N=27gsZ1z13p3j&No=96&Nrpp=96')

# Extract all html link references for webpage
# Wait 5 seconds for page to load before extracting them
time.sleep(5)
item_list = driver.find_elements_by_xpath("/html/body/div[1]/div[6]/div[2]/div[2]/div[6]/div/div/ul//div[contains(@class, 'productQvContainer')]/a[@href]")
# Note from //div[contains] is the wildcard part that selects each individual product link

# Convert selenium refs info to href links and store them in a vector
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

# Iterate over links from webpage of products to extract text data
for link in product_links:
    # wait 5 seconds before going to next link
    time.sleep(5)
    driver.get(link)
    # wait 5 seconds before scraping elements of webpage
    # e.g. allow it to load
    time.sleep(5)
    # Brand name
    brand_name = driver.find_elements_by_xpath("//*[@id='js-mobileBody']/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[1]/a[@class]")[0].text
    brand_names.append(brand_name)
    # Product name
    prod_name = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[2]")[0].text
    prod_names.append(prod_name)
    # Product size
    # If else variant for whether product is one size or has multiple sizes
    if driver.find_elements_by_xpath("//*[contains(@class,'ProductDetail__productVariantOptions')]"):
        prod_size = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[3]/div/div[1]/div[2]/span[@class='Text Text--body-2 Text--left Text--small']")[0].text
    else:
        prod_size = driver.find_elements_by_xpath("//html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[1]/p[1]")[0].text
    # Another option to print all sizes for product variant options
    # In reserve if above if statement does not work
    # driver.find_elements_by_xpath("//*[contains(@class,'ProductDetail__productVariantOptions')]")[0].text
    prod_sizes.append(prod_size)    
    # Product price
    prod_price = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/div[contains(@class, 'ProductPricingPanel')]")[0].text
    prod_prices.append(prod_price)
    # Product details
    prod_detail = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[1]/div/div")[0].text
    prod_details.append(prod_detail)
    # Product ingredients
    try:
        prod_ingredientlist = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[2]/div/div[3]/div[2]/div[2]/div/div/div")[0].get_attribute("innerText")
    except (NoSuchElementException, IndexError): 
        prod_ingredientlist = math.nan
    prod_ingredientlists.append(prod_ingredientlist)
    # Product average rating
    # For this element and below, use webdriverwait to ensure elements have loaded
    # Include try and except for new products that don't have reviews
    try:
        WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))] | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[5]/div/div[2]/div[3]/div/section/header/section/div/div[1]/div/div[1]/div/div[2]")))
        prod_rating = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-rating-decimal', ' ' ))] | /html/body/div[1]/div[4]/div/div/div/div/div/div/section[5]/div/div[2]/div[3]/div/section/header/section/div/div[1]/div/div[1]/div/div[2]").text
    except (NoSuchElementException,TimeoutException): 
        prod_rating = math.nan
    prod_ratings.append(prod_rating)
    # Product proportion of respondants who would recommend product to friends
    try:
        WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]")))
        prod_respondrec = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-reco-value', ' ' ))]").text
    except (NoSuchElementException,TimeoutException): 
        pass
    finally:
        prod_respondrec = math.nan
    prod_respondrecs.append(prod_respondrec)
    # Product total number of reviews
    try:
        WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH, "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]")))
        prod_reviewtotal = driver.find_element_by_xpath("//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'pr-snippet-review-count', ' ' ))]").text
    except (NoSuchElementException,TimeoutException): 
        prod_reviewtotal = math.nan
    prod_reviewtotals.append(prod_reviewtotal)

# Create lists for 'use_category', 'use_subcategory', 'skintype', page
use_categories = []
for string in range(len(prod_names)):
    use_categories.append('cleanser')

use_subcategory = []
for string in range(len(prod_names)):
    use_subcategory.append('face wash')

skintype = []
for string in range(len(prod_names)):
    skintype.append('oily')

page = []
for string in range(len(prod_names)):
    page.append(2)

# Combine product info lists into dataframe and export as CSV for pandas processing
df_pg2 = (pd.DataFrame(columns=['use_category', 'use_subcategory', 'skintype',
                            'brand','product','size', 'price', 'details', 
                            'ingredients', 'ratings', 'perc_respondrec', 'total_reviews', 'link', 'page'])) # creates master dataframe 

# list of each ingredient with ratings and categories paired
data_tuples = (list(zip(use_categories[1:],use_subcategory[1:],
                        skintype[1:], brand_names[1:],prod_names[1:],
                        prod_sizes[1:], prod_prices[1:], prod_details[1:],
                        prod_ingredientlists[1:], prod_ratings[1:],
                        prod_respondrecs[1:], prod_reviewtotals[1:],
                        product_links[1:], page[1:]))) 

# Create dataframe of tuple lists
temp_df = (pd.DataFrame(data_tuples,
                        columns=['use_category', 'use_subcategory', 'skintype',
                                 'brand','product','size', 'price', 'details', 
                                 'ingredients', 'ratings', 'perc_respondrec', 'total_reviews', 'link', 'page'])) # creates dataframe of each tuple in list
df_pg2 = df_pg2.append(temp_df)

driver.close()

# Export to csv
df_pg2.to_csv(f"{path}/data/cleansers_face-wash_oilyskin_pg2.csv")

# -----------
# Page 3 of 3
# ----------- 

# Start up chrome driver
driver = webdriver.Chrome(f"{path}/chromedriver")

# Fetch product page link, pg 3
driver.get('https://www.ulta.com/skin-care-cleansers-face-wash?N=27gsZ1z13p3j&No=192&Nrpp=96')

# Extract all html link references for webpage
# Wait 5 seconds for page to load before extracting them
time.sleep(5)
item_list = driver.find_elements_by_xpath("/html/body/div[1]/div[6]/div[2]/div[2]/div[6]/div/div/ul//div[contains(@class, 'productQvContainer')]/a[@href]")
# Note from //div[contains] is the wildcard part that selects each individual product link

# Convert selenium refs info to href links and store them in a vector
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

# Iterate over links from webpage of products to extract text data
for link in product_links:
    # wait 5 seconds before going to next link
    time.sleep(5)
    driver.get(link)
    # wait 3 seconds before scraping elements of webpage
    # e.g. allow it to load
    time.sleep(3)
    # Brand name
    brand_name = driver.find_elements_by_xpath("//*[@id='js-mobileBody']/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[1]/a[@class]")[0].text
    brand_names.append(brand_name)
    # Product name
    prod_name = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[2]/div/h1/div[2]")[0].text
    prod_names.append(prod_name)
    # Product size
    prod_size = driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[3]/div/div[1]/div[2]/span[2]")[0].text
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
df.to_csv(f"{path}/data/cleansers_face-wash_oilyskin_pg3.csv")











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

driver.find_elements_by_xpath("/html/body/div[1]/div[4]/div/div/div/div/div/div/section[1]/div[3]/div/div[1]/div[2]/span[2]")[0].text


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
