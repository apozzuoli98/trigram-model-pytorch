import torch
import torch.nn.functional as F

class Trigram:
    """
    Trigram Language Model
    Learns on set of names.txt
    Uses two characters to predict next character in the name

    When run, the program learns on names.txt and then prints out a sample of
    20 names

     Author: Andrew Pozzuoli
    """

    def __init__(self):
        words = open('names.txt', 'r').read().splitlines()

        chars = sorted(list(set(''.join(words))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        itos = {i:s for s,i in stoi.items()}

        # create the dataset
        xs, ys = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]
                xs.append((ix1, ix2))
                ys.append(ix3)
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        num = xs.nelement()

        # initialize the network
        g = torch.Generator()
        W = torch.randn((27, 27), generator=g, requires_grad=True)

        xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
        xenc1 = [0] * len(xenc)
        # multi-hot encoding for 2 char input
        for i in range(len(xenc)):
            xenc1[i] = xenc[i][0] + xenc[i][1]
            # for n in range(len(xenc1[i])):
            #     if xenc1[i][n].item() > 1.0:
            #         xenc1[i][n] = 1.0
        xenc = torch.stack(xenc1)

        # gradient descent
        for k in range(100):
            # forward pass
            logits = xenc @ W # predict log counts
            counts = logits.exp() # counts
            probs = counts / counts.sum(1, keepdims=True) # probabilities
            loss = -probs[torch.arange(len(xenc)), ys].log().mean() + 0.01*(W**2).mean()
            print(loss.item())

            # backward pass
            W.grad = None # set the gradient to zero
            loss.backward()

            # update
            W.data += -10 * W.grad

        print('\n')

        # sample from the model
        g = torch.Generator()
        for i in range(20):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdims=True)

                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(itos[ix])
                if ix == 0:
                    break
            print(''.join(out))




if __name__ == '__main__':
    t = Trigram()