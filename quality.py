from utils import read_classification_from_file
from confmat import BinaryConfusionMatrix
import os

SPAM_TAG = 'SPAM'
HAM_TAG = 'OK'

matrix = BinaryConfusionMatrix(SPAM_TAG, HAM_TAG)

def quality_score(tp, tn, fp, fn):
    quality = (tp + tn)/(tp + tn + 10*fp + fn)

    return quality

def compute_quality_for_corpus(corpus_dir):
    truth_path = os.path.join(corpus_dir, '!truth.txt')
    pred_path = os.path.join(corpus_dir, '!prediction.txt')

    truth_dict = read_classification_from_file(truth_path)
    pred_dict = read_classification_from_file(pred_path)

    matrix.tp = 0
    matrix.tn = 0
    matrix.fp = 0
    matrix.fn = 0

    matrix.compute_from_dicts(truth_dict, pred_dict)

    qual = quality_score(matrix.tp, matrix.tn, matrix.fp, matrix.fn)

    return qual
