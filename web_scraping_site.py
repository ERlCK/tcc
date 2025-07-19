from songs_spotify_playlist import songs
from bs4 import BeautifulSoup
import requests
import unicodedata

lyrics_link = requests.get("https://www.letras.mus.br/letodie/mais_acessadas.html").content
soup = BeautifulSoup(lyrics_link, 'html.parser')

def fixing_music_names():
    for i in range(len(songs)):
        songs[i] = songs[i].replace(" ", "-").lower()

        #Remove os acentos
        songs[i] = unicodedata.normalize("NFKD", songs[i])
        without_accents = ''
        for character in songs[i]:
            if not unicodedata.combining(character):
                without_accents = without_accents + character
        songs[i] = without_accents

def getting_letodie_links():
    letodie_links = []
    all_links = soup.find_all('a')
    for link in all_links:
        links = (link.get('href'))
        if "letodie" in links:
            letodie_links.append(links)
    return letodie_links

def getting_letodie_songs_links(letodie_links):
    songs_links = []
    for i in range(len(songs)): 
        for j in range(len(letodie_links)): 
            if songs[i] in letodie_links[j]:
                songs_links.append(letodie_links[j])
    return songs_links

fixing_music_names()
letodie_links = getting_letodie_links()
songs_links = getting_letodie_songs_links(letodie_links)
print(songs_links)
