from songs_spotify_playlist import songs
from bs4 import BeautifulSoup
import requests
import unicodedata

site_link_web = requests.get("https://www.letras.mus.br/letodie/mais_acessadas.html").content
soup_site_link = BeautifulSoup(site_link_web, 'html.parser')

lyrics_link = "https://www.letras.mus.br"

def fixing_music_names():
    for i in range(len(songs)):
        songs[i] = songs[i].replace(" ", "-").lower()

        #Remove accents
        songs[i] = unicodedata.normalize("NFKD", songs[i])
        without_accents = '' 
        for character in songs[i]:
            if not unicodedata.combining(character):
                without_accents = without_accents + character
        songs[i] = without_accents

def getting_all_letodie_links():
    letodie_links = [] #All links that contain "letodie"
    all_links = soup_site_link.find_all('a')
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

    for i in range(len(songs_links)):
        songs_links[i] = lyrics_link + songs_links[i]

    return songs_links #ex: 'https://www.letras.mus.br/letodie/memento-mori/'

def getting_songs_lyrics():
    lyrick_link_web = requests.get(songs_links[0]).content
    soup_lyric_link = BeautifulSoup(lyrick_link_web, 'html.parser')
    songs_lyrics = soup_lyric_link.find(class_='lyric-original')
    return songs_lyrics

def fixing_songs_lyrics(songs_lyrics):
    final_lyrics = str(songs_lyrics)
    final_lyrics = final_lyrics.replace('<div class="lyric-original">', "")
    final_lyrics = final_lyrics.replace("<p>", "\n")
    final_lyrics = final_lyrics.replace("</p>","")
    final_lyrics = final_lyrics.replace("<br/>","\n")
    return final_lyrics

fixing_music_names()
letodie_links = getting_all_letodie_links()
songs_links = getting_letodie_songs_links(letodie_links)
songs_lyrics = getting_songs_lyrics()
final_lyrics = fixing_songs_lyrics(songs_lyrics)
print(final_lyrics)
