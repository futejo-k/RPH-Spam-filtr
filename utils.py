def read_classification_from_file(input):
    with open(input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dictionary = {}
        for line in lines:
            l = line.strip().split()
            email = l[0]
            label = l[1]
            dictionary[email] = label

    return dictionary

def write_classification_to_file(dictionary, output):
    with open(output, 'w', encoding='utf-8') as f:
        for email, label in dictionary.items():
            f.write(email + ' ' + label + '\n')


if __name__ == '__main__':
    dict = read_classification_from_file('!truth.txt')
    write_classification_to_file(dict, '!prediction.txt')