from songs_spotify_playlist import songs
from bs4 import BeautifulSoup
import requests

lyrics_link = requests.get("https://www.letras.mus.br/letodie/mais_acessadas.html").content
soup = BeautifulSoup(lyrics_link, 'html.parser')

links = soup.find_all('a')
for link in links:
    print(link.get('href'))

