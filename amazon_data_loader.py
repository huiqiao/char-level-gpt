import pandas as pd 

def clean_dataset(text):
    EOS = '^' # begin and end token
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " ,.!?()[]{};:'\"-+*/=@#$%^&_|<>`~"
    ) 
    # add next line character and EOS char to every review
    return f"{EOS}{''.join(c for c in text if c in allowed_chars)}\n{EOS}"

def load_amazon_dataset():
    # https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data
    df = pd.read_csv('Reviews.csv')
    reviews = df['Text'].tolist()
    reviews_cleanup = [clean_dataset(review) for review in reviews]
    return reviews_cleanup


"""
# char frequency stats
char_freq = {char: 0 for char in chars}
for review in reviews_cleanup:
    for char in review:
        if char in char_freq:
            char_freq[char] += 1

for char, freq in char_freq.items():
    print(f"'{char}': {freq}")
"""