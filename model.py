import json
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import os
import pickle
import tensorflow as tf
import tflearn

nltk.download("stopwords")
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

channel_intent = {"intents": []}
channels = []
for channel in os.listdir("channels"):
    if channel.endswith(".json"):
        with open(f"channels/{channel}") as file:
            intents = json.load(file)["intents"]
            for x in intents:
                x["channel"] = channel[:-5]
            channel_intent["intents"].extend(intents)
            channels.append(channel[:-5])

model = {}
export = {}
if not os.path.exists("data"):
    os.makedirs("data")
try:
    with open(f"data/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    words_x = []
    words_y = []
    for intent in channel_intent["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            words_x.append(wrds)
            words_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [
        stemmer.stem(word.lower())
        for word in words
        if word != "?" and "'" not in word and word not in stop_words
    ]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    empty_output = [0] * len(labels)

    for x, doc in enumerate(words_x):
        total = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                total.append(1)
            else:
                total.append(0)

        output_row = empty_output[:]
        output_row[labels.index(words_y[x])] = 1

        training.append(total)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open(f"data/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

if os.path.exists("data/data.tflearn.index"):
    model.load(f"data/data.tflearn")
else:
    model.fit(
        training, output, n_epoch=10000, shuffle=True, show_metric=True, batch_size=20
    )
    model.save(f"data/data.tflearn")
export = {
    "words": words,
    "labels": labels,
    "data": channel_intent,
    "model": model,
    "channels": channels,
}


def test_words(inp, words):
    test = [0] * len(words)

    input_words = nltk.word_tokenize(inp)
    input_words = [stemmer.stem(word.lower()) for word in input_words]

    for input_word in input_words:
        for i, word in enumerate(words):
            if input_word == word:
                test[i] = 1

    return np.array(test)
