## AutoGuru

This is a working example of using NTLK with DNN for NLP for commands in a discord server.

As AutoGuru has been a meme of the Riot Games API Discord this was my basic take on making something to answer the repetitive questions.

![AutoGuru Demonstration GIF](https://www.bangingheads.net/autoguru.gif)
The discord bot supports ignoring different roles and users in settings. This is useful so it doesn't respond to people commonly answering questions.

It also supports different commands for different channels, in which if it is the top result it will refer them to that channel instead of responding.

This uses tensorflow which requires does not support python 3.8 or higher.
```pip install -r requirements.txt```

Make sure to configure settings.json with your token from the discord developers bot section.
