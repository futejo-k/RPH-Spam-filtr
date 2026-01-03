import os
import math
import re
from collections import defaultdict
from corpus import *

class MyFilter:
    def __init__(self):
        self.spam_counts = defaultdict(int)
        self.ham_counts = defaultdict(int)

        self.spam_words = 0
        self.ham_words = 0

        self.spam_mails = 0
        self.ham_mails = 0

        self.vocabulary = set()
        self.trained = False

        # required fallback if train() not called
        self.fallback_words = {
            "free", "win", "winner", "money", "credit",
            "offer", "click", "buy", "cheap", "prize"
        }

    # -------------------------------------------------
    # Text processing
    # -------------------------------------------------

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", " httpaddr ", text)
        text = re.sub(r"\S+@\S+", " emailaddr ", text)
        text = re.sub(r"\d+", " number ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    def train(self, train_corpus_dir):
        corpus = Corpus(train_corpus_dir)

        # load truth labels
        truth = {}
        with open(os.path.join(train_corpus_dir, "!truth.txt"), encoding="utf-8") as f:
            for line in f:
                fname, label = line.strip().split()
                truth[fname] = label

        for fname, body in corpus.emails():
            tokens = self._tokenize(body)

            if truth.get(fname) == "SPAM":
                self.spam_mails += 1
                for t in tokens:
                    self.spam_counts[t] += 1
                    self.spam_words += 1
                    self.vocabulary.add(t)
            else:
                self.ham_mails += 1
                for t in tokens:
                    self.ham_counts[t] += 1
                    self.ham_words += 1
                    self.vocabulary.add(t)

        self.trained = True

    # -------------------------------------------------
    # Classification
    # -------------------------------------------------

    def _classify_bayes(self, tokens):
        vocab_size = len(self.vocabulary)

        log_spam = math.log(self.spam_mails / (self.spam_mails + self.ham_mails))
        log_ham = math.log(self.ham_mails / (self.spam_mails + self.ham_mails))

        for t in tokens:
            log_spam += math.log(
                (self.spam_counts[t] + 1) / (self.spam_words + vocab_size)
            )
            log_ham += math.log(
                (self.ham_counts[t] + 1) / (self.ham_words + vocab_size)
            )

        return log_spam > log_ham

    def _classify_fallback(self, tokens):
        hits = sum(1 for t in tokens if t in self.fallback_words)
        return hits >= 2

    # -------------------------------------------------
    # Testing
    # -------------------------------------------------

    def test(self, test_corpus_dir):
        corpus = Corpus(test_corpus_dir)
        output = os.path.join(test_corpus_dir, "!prediction.txt")

        with open(output, "w", encoding="utf-8") as out:
            for fname, body in corpus.emails():
                tokens = self._tokenize(body)

                if self.trained:
                    spam = self._classify_bayes(tokens)
                else:
                    spam = self._classify_fallback(tokens)

                label = "SPAM" if spam else "OK"
                out.write(f"{fname} {label}\n")