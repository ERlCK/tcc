from songs_spotify_playlist import songs
from bs4 import BeautifulSoup
import requests
import unicodedata

site_link_web = requests.get("https://www.letras.mus.br/letodie/mais_acessadas.html").content
soup_site_link = BeautifulSoup(site_link_web, 'html.parser')

lyrics_link = "https://www.letras.mus.br"
<<<<<<< HEAD
FOLDER_NAME = "Letras/"

# deixa os nomes padronizados pra poder pegar os links corretos
=======

>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
def fixing_music_names():
    for i in range(len(songs)):
        songs[i] = songs[i].replace(" ", "-").lower()

<<<<<<< HEAD
        # remove acentos
=======
        #Remove accents
>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
        songs[i] = unicodedata.normalize("NFKD", songs[i])
        without_accents = '' 
        for character in songs[i]:
            if not unicodedata.combining(character):
                without_accents = without_accents + character
        songs[i] = without_accents

<<<<<<< HEAD
# pega todos os links do site que possuem o nome do artista
def getting_all_letodie_links():
    letodie_links = [] # todos os links que contém: "letodie"
=======
def getting_all_letodie_links():
    letodie_links = [] #All links that contain "letodie"
>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
    all_links = soup_site_link.find_all('a')
    for link in all_links:
        links = (link.get('href'))
        if "letodie" in links:
            letodie_links.append(links)
    return letodie_links

<<<<<<< HEAD
# pega todos os links das músicas do artista
=======
>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
def getting_letodie_songs_links(letodie_links):
    songs_links = [] 
    for i in range(len(songs)): 
        for j in range(len(letodie_links)): 
            if songs[i] in letodie_links[j]:
                songs_links.append(letodie_links[j])

    for i in range(len(songs_links)):
        songs_links[i] = lyrics_link + songs_links[i]

    return songs_links #ex: 'https://www.letras.mus.br/letodie/memento-mori/'

<<<<<<< HEAD
# salva as letras nos txts
=======
>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
def getting_songs_lyrics():
    for i in range(len(songs_links)):
        lyrick_link_web = requests.get(songs_links[i]).content
        soup_lyric_link = BeautifulSoup(lyrick_link_web, 'html.parser')
        songs_lyrics = soup_lyric_link.find(class_='lyric-original')
        final_lyrics = fixing_songs_lyrics(songs_lyrics)
<<<<<<< HEAD
        with open(f"{FOLDER_NAME}{songs[i]}.txt", "w", encoding="utf-8") as t:
            t.write(final_lyrics)
        print(f"Saving lyrics for {songs[i]}")
    return final_lyrics

# remove as partes desnecessárias das letras
=======
        with open(f"Lyrics/{songs[i]}.txt", "w", encoding="utf-8") as t:
            t.write(final_lyrics)
        print(f"Writing {songs[i]}")
    return final_lyrics

>>>>>>> b933669690ca5ae921dc80c296d0c9e87fc4fb9c
def fixing_songs_lyrics(songs_lyrics):
    final_lyrics = str(songs_lyrics)
    final_lyrics = final_lyrics.replace('<div class="lyric-original">', "")
    final_lyrics = final_lyrics.replace("<p>", "\n")
    final_lyrics = final_lyrics.replace("</p>","")
    final_lyrics = final_lyrics.replace("<br/>","\n")
    final_lyrics = final_lyrics.replace("</div>", "")
    return final_lyrics

fixing_music_names()
letodie_links = getting_all_letodie_links()
songs_links = getting_letodie_songs_links(letodie_links)
final_lyrics = getting_songs_lyrics()
