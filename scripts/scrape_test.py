"""
Module for scraping skincare product data from the web. 
"""

import requests

URL = 'https://www.paulaschoice.com/ingredient-dictionary'
page = requests.get(URL)

