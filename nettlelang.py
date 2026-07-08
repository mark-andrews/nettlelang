"""
nettlelang: minimal utilities for demonstrating language models.

A companion to nettle, but entirely independent of it. Nothing here
imports from or modifies nettle.

The design rule is the same as nettle's: every function is a thin,
readable wrapper whose source code is itself teaching material.

Requires torch, matplotlib, and transformers. The transformers
dependency is deliberately not declared in the package metadata,
so install it explicitly:

    pip install transformers

Typical usage:

    import nettlelang as nl

    # A pretrained model
    gpt2 = nl.load_lm("gpt2")
    nl.show_tokens(gpt2, "Unbelievable happenings in Nottingham")
    nl.plot_next_token(gpt2, "The cat sat on the")
    print(nl.generate(gpt2, "Once upon a time", temperature=0.8))
    nl.plot_attention(gpt2, "The dog chased the ball because it was tired")

    # A character-level model trained from scratch
    data = nl.load_text("shakespeare")
    model = nl.build_char_lm(data)
    print(nl.generate(model, "ROMEO:"))      # gibberish
    trainer = nl.Trainer(model)
    trainer.fit(data, steps=1000)
    print(nl.generate(model, "ROMEO:"))      # Shakespeare-shaped
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config,
                          GPT2LMHeadModel)

__version__ = "0.0.1"

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


# ---------------------------------------------------------------------------
# Tokenizers and data
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Maps characters to integer ids and back. The simplest tokenizer."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class CharData:
    """A text corpus encoded as a single long sequence of character ids."""

    def __init__(self, text):
        self.text = text
        self.tokenizer = CharTokenizer(text)
        self.ids = torch.tensor(self.tokenizer.encode(text))

    def __repr__(self):
        return (f"CharData: {len(self.text):,} characters, "
                f"vocabulary of {len(self.tokenizer)}")


def load_text(source="shakespeare", cache="tinyshakespeare.txt"):
    """Load a text corpus for character-level modelling.

    `source` may be "shakespeare", a URL, or a local file path.
    The Shakespeare corpus is cached locally after the first download.
    """
    if source == "shakespeare":
        if not os.path.exists(cache):
            urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
        source = cache
    if source.startswith("http"):
        text = urllib.request.urlopen(source).read().decode()
    else:
        with open(source, encoding="utf-8") as f:
            text = f.read()
    return CharData(text)


# ---------------------------------------------------------------------------
# The model wrapper
# ---------------------------------------------------------------------------

class LanguageModel:
    """A model together with its tokenizer, behind a uniform interface.

    Wraps either a pretrained Hugging Face model or one of our small
    character-level models. Everything else in this module works
    identically on both. The underlying torch model remains available
    at `.model` for anyone who wants to look beneath the abstraction.
    """

    def __init__(self, model, tokenizer, context=None, name="LanguageModel"):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context      # max sequence length the model accepts
        self.name = name
        self.model.eval()

    def logits(self, ids):
        """Run the model forward. Returns logits of shape (batch, seq, vocab)."""
        out = self.model(ids)
        return out.logits if hasattr(out, "logits") else out

    def n_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self):
        return (f"{self.name}: {self.n_parameters():,} parameters, "
                f"vocabulary of {len(self.tokenizer)}")


def load_lm(name="gpt2"):
    """Load a pretrained language model, e.g. "gpt2" or "distilgpt2"."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="eager")   # eager so attention is returned
    context = (getattr(model.config, "max_position_embeddings", None)
               or getattr(model.config, "n_positions", None))
    return LanguageModel(model, tokenizer, context=context, name=name)


def build_char_lm(data, layers=2, heads=2, width=128, context=256):
    """Build a small untrained GPT that reads and writes single characters."""
    config = GPT2Config(vocab_size=len(data.tokenizer), n_positions=context,
                        n_embd=width, n_layer=layers, n_head=heads)
    return LanguageModel(GPT2LMHeadModel(config), data.tokenizer,
                         context=context, name="char-transformer")


class CharRNN(nn.Module):
    """A small LSTM language model, for comparison with the transformer."""

    def __init__(self, vocab_size, width=128, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.lstm = nn.LSTM(width, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, ids):
        h, _ = self.lstm(self.embed(ids))
        return self.head(h)


def build_char_rnn(data, layers=2, width=128, hidden=256):
    """Build a small untrained LSTM that reads and writes single characters."""
    model = CharRNN(len(data.tokenizer), width, hidden, layers)
    return LanguageModel(model, data.tokenizer, context=None, name="char-rnn")


# ---------------------------------------------------------------------------
# Demo 1: tokens
# ---------------------------------------------------------------------------

def show_tokens(lm, text):
    """Print the tokens a piece of text is split into."""
    for i in lm.tokenizer.encode(text):
        print(f"{i:>6}  {lm.tokenizer.decode([i])!r}")


# ---------------------------------------------------------------------------
# Demo 2: the next-token distribution
# ---------------------------------------------------------------------------

def plot_next_token(lm, prompt, top_k=10):
    """Bar chart of the model's probabilities for the next token."""
    ids = torch.tensor([lm.tokenizer.encode(prompt)])
    with torch.no_grad():
        probs = F.softmax(lm.logits(ids)[0, -1], dim=-1)
    top = torch.topk(probs, top_k)
    labels = [repr(lm.tokenizer.decode([i])) for i in top.indices]
    plt.figure()
    plt.barh(labels[::-1], top.values.flip(0).numpy())
    plt.xlabel("probability")
    plt.title(f"Next token after {prompt!r}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo 3: sentence probability
# ---------------------------------------------------------------------------

def perplexity(lm, text):
    """How surprised the model is by a piece of text, as a single number.

    Demo 2 asked what the model expects one token ahead. This asks the
    same question for a whole passage at once: how well does the model's
    next-token distribution, applied over and over, predict text that
    actually occurs? Perplexity is the exponential of the average
    per-token cross-entropy loss, so a value of k reads as "on average,
    as uncertain as choosing among k equally likely tokens". Lower means
    more expected, more fluent, more probable under the model. Compare a
    fluent sentence against a scrambled version of the same words.
    """
    ids = torch.tensor([lm.tokenizer.encode(text)])
    with torch.no_grad():
        logits = lm.logits(ids)[0, :-1]
        targets = ids[0, 1:]
        loss = F.cross_entropy(logits, targets)
    return loss.exp().item()


# ---------------------------------------------------------------------------
# Demo 4: generation is repeated sampling from that distribution
# ---------------------------------------------------------------------------

def generate(lm, prompt, n=100, temperature=0.8, seed=None):
    """Generate a continuation of the prompt, one token at a time.

    Temperature scales the logits before sampling. Low values are
    conservative and repetitive, high values degenerate into noise.
    A temperature of zero always takes the single most likely token.
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = torch.tensor([lm.tokenizer.encode(prompt)])
    with torch.no_grad():
        for _ in range(n):
            window = ids[:, -lm.context:] if lm.context else ids
            logits = lm.logits(window)[0, -1]
            if temperature <= 0:
                next_id = logits.argmax().view(1, 1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).view(1, 1)
            ids = torch.cat([ids, next_id], dim=1)
    return lm.tokenizer.decode(ids[0].tolist())


# ---------------------------------------------------------------------------
# Demo 5: attention
# ---------------------------------------------------------------------------

def plot_attention(lm, sentence, layer=5, head=8):
    """Heatmap of one attention head. Each row shows where that token looks.

    Only transformers have attention, so this raises an error for the RNN.
    Most heads attend to boring things. Find a good layer and head at home
    and hard-code them for the lecture.
    """
    if isinstance(lm.model, CharRNN):
        raise ValueError(f"{lm.name} is an RNN and has no attention weights")
    ids = torch.tensor([lm.tokenizer.encode(sentence)])
    with torch.no_grad():
        out = lm.model(ids, output_attentions=True)
    att = out.attentions[layer][0, head]
    tokens = [lm.tokenizer.decode([i]) for i in ids[0]]
    plt.figure()
    plt.imshow(att, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"{lm.name}, layer {layer}, head {head}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def compare_attention(lm, sentence, heads):
    """Side-by-side heatmaps for several (layer, head) pairs, one sentence.

    Most heads attend to boring things, a handful do something a human
    would recognise. Seeing several side by side, rather than one at a
    time, is what makes that contrast visible: same sentence, same
    model, wildly different behaviour, purely from which head is
    looking. `heads` is a list of (layer, head) tuples.
    """
    if isinstance(lm.model, CharRNN):
        raise ValueError(f"{lm.name} is an RNN and has no attention weights")
    ids = torch.tensor([lm.tokenizer.encode(sentence)])
    with torch.no_grad():
        out = lm.model(ids, output_attentions=True)
    tokens = [lm.tokenizer.decode([i]) for i in ids[0]]
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 5))
    for ax, (layer, head) in zip(axes, heads):
        att = out.attentions[layer][0, head]
        ax.imshow(att, cmap="viridis")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_title(f"layer {layer}, head {head}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo 6: word embeddings
# ---------------------------------------------------------------------------

def _token_id(lm, word):
    """Encode `word` to a single token id, or raise a clear error.

    A leading space is added, since that is how a word is tokenized when
    it is not the first word in a sentence, which is the common case.
    """
    ids = lm.tokenizer.encode(" " + word)
    if len(ids) != 1:
        pieces = [lm.tokenizer.decode([i]) for i in ids]
        raise ValueError(f"{word!r} is not a single token, it splits into "
                         f"{pieces}. Try a shorter or more common word.")
    return ids[0]


def _nearest(lm, vectors, target, k, exclude):
    """The k rows of `vectors` most similar to `target`, by cosine similarity."""
    sims = F.cosine_similarity(target.unsqueeze(0), vectors)
    for i in exclude:
        sims[i] = -1.0
    top = torch.topk(sims, k)
    return [(lm.tokenizer.decode([i]), v.item()) for v, i in zip(top.values, top.indices)]


def nearest_tokens(lm, word, k=10):
    """Tokens whose embeddings are closest to `word`'s, by cosine similarity.

    Every token has a fixed row in the model's input embedding table,
    learned from which contexts it tends to appear in, before any
    attention or context is applied. Nearby vectors tend to be nearby in
    meaning, in inflection (plurals, capitalisation), or both. Only
    meaningful for a trained transformer, so this is for `load_lm`
    models, not the untrained toy ones below.
    """
    if isinstance(lm.model, CharRNN):
        raise ValueError(f"{lm.name} has no token embedding table")
    vectors = lm.model.transformer.wte.weight
    i = _token_id(lm, word)
    return _nearest(lm, vectors, vectors[i], k, exclude={i})


def analogy(lm, a, b, c, k=5):
    """`a` is to `b` as `c` is to ...

    The classic demonstration that embeddings encode relationships as
    directions, not just points: the vector from `a` to `b`, added to
    `c`, tends to land near the word completing the same relationship.
    `analogy(lm, "man", "king", "woman")` lands near "queen". Not
    guaranteed for any given triple, and only as good as the embedding
    table it is drawn from.
    """
    if isinstance(lm.model, CharRNN):
        raise ValueError(f"{lm.name} has no token embedding table")
    vectors = lm.model.transformer.wte.weight
    ia, ib, ic = _token_id(lm, a), _token_id(lm, b), _token_id(lm, c)
    vector = vectors[ib] - vectors[ia] + vectors[ic]
    return _nearest(lm, vectors, vector, k, exclude={ia, ib, ic})


# ---------------------------------------------------------------------------
# Demo 7: toy training
# ---------------------------------------------------------------------------

class Trainer:
    """Trains a LanguageModel on a CharData corpus by next-token prediction.

    The same grammar as nettle's Trainer: construct with a model,
    then call fit with the data. Works identically for the transformer
    and the RNN, which is rather the point.
    """

    def __init__(self, lm, lr=3e-4):
        self.lm = lm
        self.optimizer = torch.optim.AdamW(lm.model.parameters(), lr=lr)
        self.losses = []

    def fit(self, data, steps=1000, batch_size=32, block_size=128,
            report_every=100):
        self.lm.model.train()
        for step in range(steps):
            # a random batch of text snippets
            starts = torch.randint(len(data.ids) - block_size - 1,
                                   (batch_size,))
            x = torch.stack([data.ids[i:i + block_size] for i in starts])
            y = torch.stack([data.ids[i + 1:i + block_size + 1]
                             for i in starts])
            # the objective: predict y, which is x shifted one step ahead
            logits = self.lm.logits(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            if step % report_every == 0 or step == steps - 1:
                print(f"step {step:>5}   loss {loss.item():.3f}")
        self.lm.model.eval()
        return self

    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.title(f"Training {self.lm.name}")
        plt.tight_layout()
        plt.show()
