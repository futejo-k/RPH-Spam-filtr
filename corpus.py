import os

class Corpus:
    def __init__(self, path):
        self.path = path

    def emails(self):
        for fname in os.listdir(self.path):
            if fname.startswith('!'):
                continue
            
            path_to_file = os.path.join(self.path, fname)

            if not os.path.isfile(path_to_file):
                continue

            with open(path_to_file, 'r', encoding='utf-8') as f:
                body = f.read()
            
            yield fname, body

if __name__ == "__main__":
    corpus = Corpus("./emails")
    count = 0
    for fname, body in corpus.emails():
        print(fname)
        print(body)
        print('-------------------------')
        count+=1
    print('Finished: ', count, 'files processed.')
