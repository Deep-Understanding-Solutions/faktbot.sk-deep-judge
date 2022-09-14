import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.read_csv("data/scrapped.csv", index_col="id")

# Article identifiers for each page.
article_data = {
    "slovenskeslovo": {
        "total": 2670,
        "parsed": 1898
    },
    "ta3": {
        "total": 245922,
        "parsed": 2526
    }
}

# Selector to get correct quantities.
selector = "slovenskeslovo"

for article_decrement in range(article_data[selector]['total'] - article_data[selector]['parsed']):
    print(f"Articles parsed: {article_decrement + article_data[selector]['parsed']}")
    req = requests.get(f"https://slovenskeslovo.sk/{article_data[selector]['total'] - article_decrement}")

    # Parsing is too specific for each platform, so it is solved using comments.
    soup = BeautifulSoup(req.content, 'html5lib')

    try:
        # TA3
        # title = soup.find('h1', attrs={'class': 'article-title'}).get_text().strip().replace("\xa0", " ")W
        # slovenskeslovo
        title = soup.find('h2', attrs={'class': 'avatar-article-heading'}).get_text().strip().replace("\xa0", " ")
    except Exception:
        title = ""

    try:
        # TA3
        # perex = soup.find('div', attrs={'class': 'article-perex'}).get_text()
        # slovenskeslovo
        perex = ""
    except Exception:
        perex = ""

    try:
        # TA3
        # content = soup.find('div', attrs={'class': 'article-component'}).find_all('p')
        # content = list(map(lambda paragraph: paragraph.get_text().strip().replace("\xa0", " "), content))
        # slovenskeslovo
        content = soup.find('div', attrs={'class': 'item-page'}).find_all('p')
        content = list(map(lambda paragraph: paragraph.get_text().strip().replace("\xa0", " "), content))

        article = perex.join(content)
    except Exception:
        article = ""

    if title != "" and article != "":
        df.loc[len(df.index)] = [title, article, "None", "None", "None", 0]
        df.to_csv('data/scrapped.csv', mode="w", header=False)
