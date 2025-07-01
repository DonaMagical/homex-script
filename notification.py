import os
import requests

def send_push_notification(title: str, message: str, sound: str = "pushover"):
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        return

    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": token,
        "user": user,
        "title": title,
        "message": message,
        "priority": 1,
        "sound": sound
    }
    response = requests.post(url, data=data)
    return response.json()