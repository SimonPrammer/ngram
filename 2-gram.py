#%%

words = open('data/train.txt', 'r').read().splitlines()
# words[:10]
len(words)

#%%
min([len(word) for word in words]), max([len(word) for word in words])

#%%
#count how often some char follows another
b = {}

for w in words:
    #introduce special start and end
    chrs = ["<S>"] + list(w) + ["<E>"]
    for c1,c2 in zip(chrs, chrs[1:]):
        bigram = (c1,c2)
        b[bigram] = b.get(bigram, 0) + 1


#%%
sorted_b = sorted(b.items(), key=lambda item: -item[1])
sorted_b
#%%
import torch

#lookup table for the strings to int because we need this for tensor
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
# stoi["<S>"] = 26
# stoi["<E>"] = 27
stoi["."] = 0
stoi
#%%
# N = torch.zeros((28,28),dtype=torch.int32)
N = torch.zeros((27,27),dtype=torch.int32)

for w in words:
    #introduce special start and end
    # chrs = ["<S>"] + list(w) + ["<E>"]
    chrs = ["."] + list(w) + ["."]
    for c1,c2 in zip(chrs, chrs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        N[ix1,ix2] += 1 

N

#%%
import matplotlib.pyplot as plt
plt.imshow(N)
#%%
#better visualisation

itos = {i:s for s,i in stoi.items()}
itos

plt.figure(figsize=(16,16))
plt.imshow(N,cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j,i,chstr,ha="center",va="bottom",color="gray")
        plt.text(j,i,N[i,j].item(),ha="center",va="top",color="gray")

plt.axis("off")

#%%

#probability of the first token (has to start with . and then the likelihood of which token following)
p = N[0].float()
p = p / p.sum()
p

#we can sample from this with torch.multinominal which gives samples from a probability distribution
#if we use a generator it will be "deterministic"
#we can use samplers do walk through the rows of my distributions and sample letters

ix = 0
generator = torch.Generator().manual_seed(2147483647)

#take a row, sample from its probability distribution, walk through the rows until I hit end token. TADA!
for i in range(10):
    out = []
    while True:
        p = N[ix, :]
        p = p / p.sum()
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))


#%%

#we can make this more efficient by making a matrix P with the probabilities
#then we dont have to calculate it for every row.

#each row normalize the tensor with a sum that adds up each row into a [27,1] vector
ix = 0
generator = torch.Generator().manual_seed(2147483647)

# P = N.float() / N.sum(dim=1,keepdim=True)
#inplace operation is slightly faster as it does not create new memory
# +1 because of model smoothing! 
# otherwise we might have a zero probability in P which results in -inf log likelihood!!
P = (N+1).float()
P /= P.sum(dim=1,keepdim=True)

#we can check if this is properly done by summing the rows
print(P[0],P[0].sum())
#and show that the cols are not adding up to 1
print(P[:, 0].sum())
#which is exactly what is correct. if we discard keepdim it will be reversed because of broadcasting

for i in range(10):
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))

#%%

#now lets check how good it is with a loss fn
#min 52
log_likelihood = 0.0
n=0

#we can evaluate that for any word 
# for w in ["simon"]:
for w in words[:1]:
# for w in words:
    #introduce special start and end
    # chrs = ["<S>"] + list(w) + ["<E>"]
    chrs = ["."] + list(w) + ["."]
    for c1,c2 in zip(chrs, chrs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n +=1
        print(f"{c1}{c2}: {prob: .4f}:  {logprob: .4f}")
        
# print("expected prob per combination:", 1/27)

# we use maximum log likelihood as loss
log_likelihood

#we want negative log likelihood because its conform to loss fns and we can minimize it
nll = -log_likelihood
#we like to normalize the nll 
nll /= n
print(f"average negative log likelihood: {nll=}. we want to drive this to 0 by optimizing for the right parameters w.")
#we want to minimize the nll on our model (as we want to maximize likelihood)
#%%

# Neural Network approach:

# create training set of bigram (x,y)
# we are given the first character and we are trying to predict the next one of the two.
xs, ys = [], []



words = open('data/train.txt', 'r').read().splitlines()
for w in words[:1]:
    chrs = ["."] + list(w) + ["."]
    for c1,c2 in zip(chrs, chrs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        print(f"{c1}{c2}")
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xs,ys

