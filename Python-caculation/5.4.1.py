import matplotlib.pyplot as plt
from collections import Counter
import string
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
def count_words(text):
    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).split()
    word_counts = Counter(words)
    return word_counts


def choose_word_colors(word_counts):
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    max_count = max(sorted_word_counts.values())

    colors = []
    for word, count in sorted_word_counts.items():
        normalized_count = count / max_count
        color = plt.cm.RdYlBu(normalized_count)
        colors.append(color)

    word_colors = {word: color for word, color in zip(sorted_word_counts.keys(), colors)}
    return word_colors

def plot_word_frequency(word_counts, word_colors, top_n=10):
    most_common_words = word_counts.most_common(top_n)
    words, counts = zip(*most_common_words)
    colors = [word_colors[word] for word in words]

    plt.figure(figsize=(19, 8))
    bars = plt.bar(words, counts, color=colors)
    plt.xlabel('words')
    plt.ylabel('frequency')
    plt.title(f'top {top_n} words')
    plt.xticks(rotation=45)


    color_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(color_patches, words, title="words", loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "../Python-basis/test.txt"
    text = read_text_file(file_path)
    word_counts = count_words(text)
    word_colors = choose_word_colors(word_counts)
    plot_word_frequency(word_counts, word_colors, 30)
