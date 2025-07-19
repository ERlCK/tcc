from bs4 import BeautifulSoup
import requests
import codecs
import os

all_links = requests.get("https://www.letras.mus.br/letodie/mais_acessadas.html").content
soup = BeautifulSoup(all_links, 'html.parser')

filtered_links = []
lyric = ''
final_lyric = ''
files_path = []

def letodieFilter():
    for link in soup.find_all('a'):
        links = (link.get('href'))
        if links and 'letodie/' in links:
            filtered_links.append(links)

def filter():
    for i in range(6):
        filtered_links.pop(0)
    filtered_links.reverse()
    for x in range(11):
        filtered_links.pop(0)
    filtered_links.reverse()

def getLyric(link):
    songsLinks = requests.get("https://www.letras.mus.br{}".format(link)).content
    songsLinks = BeautifulSoup(songsLinks, 'html.parser')
    lyric = songsLinks.find(class_='lyric-original')
    return lyric

def replaces(lyric):
    lyric = str(lyric)
    final_lyric = lyric.replace("<br>","\n")
    final_lyric = final_lyric.replace("<br/>", "\n")
    final_lyric = final_lyric.replace("</p>", "\n\n")
    final_lyric = final_lyric.replace('<div class="lyric-original"> ', "")
    final_lyric = final_lyric.replace("</div>", "")
    final_lyric = final_lyric.replace("<p>", "")
    return final_lyric

def file(final_lyric, file_path):
    #f = open('E:\VSCODE\Projetos\TestandoAI\Langchain_ChatGPT\Arquivostxt', mode='w+')
    with codecs.open(file_path, 'w', encoding='utf-16') as f:
        f.write(str(final_lyric))
    #print(file_path)

letodieFilter()
filter()

for link in filtered_links:
    fileName = link.split("/")[2]
    lyric = getLyric(link)
    final_lyric = replaces(lyric)
    file_path = f'E:\\VSCODE\\Projetos\\TCC\\Lyrics\\{fileName}.txt'
    file(final_lyric, file_path)
    files_path.append(file_path)
    print(file_path)

#print(soup.prettify())