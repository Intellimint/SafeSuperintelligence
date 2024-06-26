import requests
from bs4 import BeautifulSoup

class CuriosityScraper:
    def __init__(self):
        pass

    def scrape_webpage(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text(separator='\n')
            return content
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def fetch_and_update_data(self, urls):
        data = {}
        for url in urls:
            content = self.scrape_webpage(url)
            data[url] = content
        return data
