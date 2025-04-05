import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from nltk.classify import MaxentClassifier
from nltk.tag import hmm

# load the data
nltk.download('treebank')
sentences = treebank.tagged_sents()
train_data, test_data = train_test_split(sentences, test_size = 0.2, random_state = 42)
words = set(word for sent in train_data for word, _ in sent)
tags = set(tag for sent in train_data for _, tag in sent)
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

#counting transition and emission probabilities
transition = defaultdict(lambda: defaultdict(int))
emission = defaultdict(lambda: defaultdict(int))
tag_count = defaultdict(int)

for sent in train_data:
    prev_tag = "<s>"
    for word, tag in sent:
        transition[prev_tag][tag] += 1
        emission[tag][word] += 1
        tag_count[tag] += 1
        prev_tag = tag

#converting to probabilities
def prob(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}

transition_prob = {k: prob(v) for k, v in transition.items()}
emission_prob = {k: prob(v) for k, v in emission.items()}


#viterbi algorithm
def viterbi(sentence):
    V = [{}]
    path = {}
    for tag in tags:
        V[0][tag] = transition_prob["<s>"].get(tag, 1e-6) * emission_prob[tag].get(sentence[0], 1e-6)
        path[tag] = [tag]

    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        for tag in tags:
            (prob, state) = max(
                    (V[t -1][prev_tag] * transition_prob[prev_tag].get(tag, 1e-6) * emission_prob[tag].get(sentence[t], 1e-6), prev_tag)
                    for prev_tag in tags
            )
            V[t][tag] = prob
            new_path[tag] = path[state] + [tag]
        path = new_path

    (prob, final_tag) = max((V[-1][tag], tag) for tag in tags)
    return path[final_tag]

#evaluate accuracy
correct = total = 0
for sent in test_data:
    words = [word for word, _ in sent]
    true_tags = [tag for _, tag in sent]
    pred_tags = viterbi(words)

    for t, p in zip(true_tags, pred_tags):
        if t == p:
            correct += 1
        total += 1

print(f"Viterbi HMM POS Tagging Accuracy: {correct / total:.4f}")

#Maximum Entropy HMM-based Tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train_supervised(train_data)
accuracy = hmm_model.evaluate(test_data)
print(f"MaxEnt HMM POS Tagging Accuracy: {accuracy:.4f}")
