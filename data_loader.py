import pandas as pd 

# get vocabulary and special character like end of sentence. Vocabulary from amazon product reviews
def get_vocab():
    chars = ''' !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\n'''
    EOS = '^' # begin and end token
    return chars, EOS 

def clean_dataset(text):
    chars, EOS = get_vocab()
    # add next line character and EOS char to every review
    return f"{EOS}{''.join(c for c in text if c in chars)}\n{EOS}"

def load_shakespeare_dataset():
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data//input.txt
    with open('tinyshakespeare_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        return clean_dataset(text)

def load_amazon_dataset():
    # https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data
    df = pd.read_csv('Reviews.csv')
    reviews = df['Text'].tolist()
    reviews_cleanup = [clean_dataset(review) for review in reviews]
    return reviews_cleanup