def load_shakespeare_dataset():
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data//input.txt
    with open('tinyshakespeare_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        return text