class BinaryConfusionMatrix:
    def __init__(self, pos_tag, neg_tag):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag

    def as_dict(self):
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}

    def check_value(self, value):
        if value not in (self.pos_tag, self, self.neg_tag):
            raise ValueError(f'The "truth" parameter can be either {self.pos_tag} or {self.neg_tag}.')

    def update(self, truth, prediction):
        self.check_value(truth)
        self.check_value(prediction)
        if prediction == self.pos_tag:
            if prediction == truth:
                self.tp += 1
            else:
                self.fp += 1
        elif prediction == self.neg_tag:
            if prediction == truth:
                self.tn += 1
            else:
                self.fn += 1

    def compute_from_dicts(self, truth_dict, pred_dict):
        for key in truth_dict:
            self.update(truth_dict[key], pred_dict[key])
