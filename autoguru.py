from model import export, test_words
import numpy as np
import discord
import json
import requests
import requests_cache

with open("settings.json") as file:
    settings = json.load(file)

requests_cache.install_cache("ddragon", backend="memory", expire_after=3600)


class AutoGuru(discord.Client):
    async def on_ready(self):
        print("READY")

    async def on_guild_available(self, guild):
        print(guild.id)

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        elif message.channel.name not in export["channels"]:
            return
        elif message.author.id in settings["ignore_members"]:
            return
        elif (
            len([x for x in message.author.roles if x in settings["ignore_roles"]]) > 0
        ):
            return
        else:
            inp = message.content
            result = export["model"].predict([test_words(inp, export["words"])])[0]
            result_index = np.argmax(result)
            tag = export["labels"][result_index]

            if result[result_index] > 0.8:
                for tg in export["data"]["intents"]:
                    if tg["tag"] == tag:
                        if (
                            tg["channel"] == message.channel.name
                            or tg["channel"] == "all"
                        ):
                            response = tg["response"]
                        else:
                            channel_id = discord.utils.get(
                                message.guild.channels, name=tg["channel"]
                            )
                            await message.channel.send(
                                f"Did you mean to send this in <#{channel_id.id}>? There is a response available for that channel.".format(
                                    message
                                )
                            )
                            return
                await message.channel.send(replace_vars(response))
            else:
                for intent in export["data"]["intents"]:
                    if intent["tag"] in inp.lower().split():
                        if (
                            intent["channel"] == message.channel.name
                            or intent["channel"] == "all"
                        ):
                            response = intent["response"]
                            await message.channel.send(replace_vars(response))
                        else:
                            channel_id = discord.utils.get(
                                message.guild.channels, name=intent["channel"]
                            )
                            await message.channel.send(
                                f"Did you mean to send this in <#{channel_id.id}>? There is a response available for that channel.".format(
                                    message
                                )
                            )
                        break


def get_ddragon_version():
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json")
    return versions.json()[0]


def replace_vars(inp):
    replaceable = {"ddragon_version": get_ddragon_version()}
    return inp.format(**replaceable)