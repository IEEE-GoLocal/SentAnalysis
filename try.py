from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model
model = load_model('rnn_model.h5')

import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

import pickle
with open("tokenizer.pkl", "rb") as input_file:
    tk = pickle.load(input_file)

testing = [
    "This product is good",
    "This product is bad",
    "Product is not satisfactory",
    "This product is okayish"
]
MAX_LENGTH = 255
test = normalize_texts(testing)
test = tk.texts_to_sequences(test)
test = pad_sequences(test, maxlen=MAX_LENGTH)
out = model.predict(test)

for i in range(len(out)):
    print(testing[i],end=" : ")
    if out[i]>=0.5:
        print("POSITIVE")
    else:
        print("NEGATIVE")