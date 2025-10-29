from dotenv import load_dotenv
import os
import base64
from requests import post
import requests
import json

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": 'Basic ' + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "client_credentials"
    }

    result = post(url, headers=headers, data=data)
    token = json.loads(result.content)
    token = token["access_token"]
    return (token)

def get_auth_header(token):
    return {
        "Authorization": "Bearer " + token
    }

def get_spotify_playlist_items(playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    params =  {
        "limit": 20,
        "offset": 0
    }
    songs = []

    while True:
        result = requests.get(url, headers=headers, params=params)
        data =  result.json()

        for item in data["items"]:
            name = item["track"]["name"]
            songs.append(name)
            
        if data["next"] is None:
            break
        
        params["offset"] += params["limit"]

    return songs

token = get_token()
songs = get_spotify_playlist_items("6s5qmXJ05wFkZS2GIioXaR")
#print(songs)