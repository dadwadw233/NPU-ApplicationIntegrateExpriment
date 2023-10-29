import string
from collections import Counter


def count_word_frequency(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).split()
    word_count = Counter(words)

    return word_count


file_path = "./test.txt"
result = count_word_frequency(file_path)

for word, count in result.items():
    print(f"{word}: {count}")
