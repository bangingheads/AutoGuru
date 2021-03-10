import json
from autoguru import AutoGuru

with open("settings.json") as file:
    settings = json.load(file)

client = AutoGuru()
client.run(settings['token'])
