import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_top_100_movies():
    url = "https://movie.douban.com/top250"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    movies = []

    for i in range(0, 100, 25):
        response = requests.get(url, headers=headers, params={"start": i})
        soup = BeautifulSoup(response.text, 'html.parser')
        movie_list = soup.find_all('div', class_='info')

        for movie in movie_list:
            title = movie.find('div', class_='hd').find('a').find('span').text
            link = movie.find('div', class_='hd').find('a')['href']
            rating = movie.find('div', class_='bd').find('div', class_='star').find('span', class_='rating_num').text
            try:
                quote = movie.find('span', class_='inq').text
            except:
                quote = ""

            movies.append([title, link, rating, quote])

    return movies


def save_to_csv(movies):
    df = pd.DataFrame(movies, columns=["Title", "Link", "Rating", "Quote"])
    df.to_csv("movies.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    movies = get_top_100_movies()
    save_to_csv(movies)
