import os
import math
import re
from collections import defaultdict

from corpus import Corpus
from utils import read_classification_from_file, write_classification_to_file


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

        self.fallback_words = {
            "free", "win", "winner", "money", "credit",
            "offer", "click", "buy", "cheap", "prize"
        }

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", " httpaddr ", text)
        text = re.sub(r"\S+@\S+", " emailaddr ", text)
        text = re.sub(r"\d+", " number ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    def train(self, train_corpus_dir):
        corpus = Corpus(train_corpus_dir)

        truth_path = os.path.join(train_corpus_dir, "!truth.txt")
        truth = read_classification_from_file(truth_path)

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

    def test(self, test_corpus_dir):
        corpus = Corpus(test_corpus_dir)
        predictions = {}

        for fname, body in corpus.emails():
            tokens = self._tokenize(body)

            if self.trained:
                spam = self._classify_bayes(tokens)
            else:
                spam = self._classify_fallback(tokens)

            predictions[fname] = "SPAM" if spam else "OK"

        output_path = os.path.join(test_corpus_dir, "!prediction.txt")
        write_classification_to_file(predictions, output_path)
