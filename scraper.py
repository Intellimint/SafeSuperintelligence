import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.prettify()
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"

