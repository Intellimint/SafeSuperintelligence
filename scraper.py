import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = soup.findAll(text=True)
            visible_texts = filter(tag_visible, texts)
            return u" ".join(t.strip() for t in visible_texts)
        else:
            return f"Error: Unable to fetch the URL (status code: {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"

def tag_visible(element):
    if isinstance(element, Comment):
        return False
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    return True

def scrape_webpage(url):
    return extract_text_from_url(url)
