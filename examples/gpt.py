import mlx.core as mx

from softgrad import Network
from softgrad.function.core import Add
from softgrad.layer.core import Parallel, Embedding


with open('rsc/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4 # reduce learning rate
eval_iters = 200
n_embd = 768


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = mx.array(encode(text), mx.int64)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))
    x = mx.stack([data[i:i + block_size] for i in ix])
    y = mx.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


network = Network(input_shape=(block_size,))

network.add_layer(Parallel([
    Embedding(vocab_size, n_embd),      # Semantic encoding
    Embedding(block_size, n_embd),      # Positional encoding
], Add()))


# def generate(self, idx, max_new_tokens):
#     # idx is (B, T) array of indices in the current context
#     for _ in range(max_new_tokens):
#         # crop idx (B, T) array of indices in the current context
#         # (never pass more than block_size tokens)
#         idx_cond = idx[:, -block_size:]
#         # get the predictions
#         logits, loss = self(idx_cond)
#         # focus only on the last time step
#         logits = logits[:, -1, :]  # becomes (B, C)
#         # apply softmax to get probabilities
#         probs = F.softmax(logits, dim=-1)  # (B, C)
#         # sample from the distribution
#         idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
#         # append sampled index to the running sequence
#         idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
#     return idx

# for iter in range(max_iters):
#
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# losses = estimate_loss()
# print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))