import time
from datetime import datetime
import torch
from gpt import GPTLanguageModel
from gpt import GPTConfig
from shakespeare_data_loader import load_shakespeare_dataset
from amazon_data_loader import load_amazon_dataset


# hyperparameters
batch_size = 64
block_size = 256 # same value with GPTConfig.block_size, not elegant
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------

# get vocabulary and special character like end of sentence. Vocabulary from amazon product reviews
def get_vocab():
    chars = ''' !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\n'''
    EOS = '^' # begin and end token
    return chars, EOS 

# generate a small batch of data of inputs x and targets y
def get_batch(data, config):
    if isinstance(data, list): # data is list of sentences, eg: amazon product reviews
        ix = torch.randint(len(data), (batch_size,))
        x_list = []
        y_list = []
        for i in ix:
            rand_int = torch.randint(0, len(data[i]) - config.block_size, (1,))
            x_list.append(data[i][rand_int : rand_int+config.block_size])
            y_list.append(data[i][rand_int+1 : rand_int+config.block_size+1])
        x = torch.stack(x_list)
        y = torch.stack(y_list)
    else: # just a tensor, list of characters
        ix = torch.randint(len(data) - config.block_size, (batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = []
    model.eval()
    for data in [train_data, val_data]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, model.config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out.append(losses.mean())
    model.train()
    return out


torch.manual_seed(1337)

text = load_shakespeare_dataset()
chars, EOS = get_vocab()

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

config = GPTConfig(vocab_size = len(chars))
model = GPTLanguageModel(config, device)
model.load_state_dict(torch.load('char_gpt2024-03-17-05-11-55.pth'))
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, train_data, val_data)
        print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, config)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# save model
torch.save(model.state_dict(), f'char_gpt{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth')

# generate from the model
context = torch.tensor([encode('^')])
c = context.to(device)
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(c, max_new_tokens=500)[0].tolist()))
