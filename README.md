# 🌪️ Disaster Tweets NLP Classifier

Can a machine tell the difference between "the whole sky is ash" (real wildfire) and "this concert was 🔥" (not a disaster)? This project tries to find out.

A text classification model that predicts whether a tweet is about a **real disaster** or not, built on Kaggle's "NLP with Disaster Tweets" competition.

## 🎯 Problem Statement
Given the raw text of a tweet, classify it as:
- `1` → real disaster 🚨
- `0` → not a disaster (probably just dramatic Twitter) 😅

Tricky part: tweets are messy, sarcastic, and full of slang — so this isn't a clean textbook dataset.

## 📊 Dataset
- Source: [Kaggle - NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started)
- ~7,600 labeled training tweets, ~3,200 test tweets
- Features: tweet `text`, `keyword`, `location` (often missing 🕳️)

## 🛠️ Approach
1. **🧹 Text cleaning** — lowercased everything, stripped URLs and punctuation
2. **🔢 Feature extraction** — TF-IDF with unigrams + bigrams, sublinear scaling, 20K max features
3. **🤖 Modeling** — Logistic Regression as the core classifier
4. **🎚️ Threshold tuning** — swept thresholds from 0.10 → 0.99 to find the sweet spot instead of blindly trusting the default 0.5 cutoff
5. **🏁 Final training** — retrained on 100% of the data with the winning threshold before predicting on test

## 📈 Results

| Model Version | Validation F1 |
|---|---|
| Default threshold (0.50) | 0.761 |
| 🏆 Tuned threshold (0.47) | **0.7659** |

**Kaggle Leaderboard Score:** *[drop your final score here]* 🎉

## 🧰 Tech Stack
Python 🐍 · Pandas · NumPy · Scikit-learn · TF-IDF · Logistic Regression

## 💡 Key Takeaway
You don't always need a fancier model — sometimes just **tuning the decision boundary** squeezes out real performance gains. Small tweak, measurable win.

## 🚀 How to Run
1. Clone this repo
2. Grab `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/nlp-getting-started/data)
3. Open the notebook and run all cells top to bottom
4. Watch it guess whether tweets are actually about disasters 🔮
