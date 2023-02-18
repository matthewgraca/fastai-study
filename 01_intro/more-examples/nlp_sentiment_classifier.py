from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=8)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

text = ["I really liked that movie!",
        "This is considered good because everyone has bad taste.",
        "People seem to think this isn't good, but I consider this a classic."
        ]
for t in text:
    print(f"Review: {t}")
    pos_or_neg,_,probs = learn.predict(t)
    print(f"This is a {pos_or_neg} review")
    print(f"Probability of pos: {probs[1]:.4f}")
    print(f"Probability of neg: {probs[0]:.4f}\n")
