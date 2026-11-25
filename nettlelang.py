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

__version__ = "0.0.2"

# Use an NVIDIA GPU when there is one. Only the classification functions
# at the end of this module move models there; everything else runs
# wherever its model already is, which is the CPU unless moved.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    """A text corpus encoded as a single long sequence of token ids.

    Defaults to a fresh character-level tokenizer built from the text
    itself, for training a model from scratch. Pass `tokenizer=`, an
    existing model's tokenizer such as `gpt2.tokenizer`, to encode with
    that instead. That is what fine-tuning a pretrained model requires:
    it must keep reading and writing the vocabulary it already knows,
    not a new one built from this one corpus.
    """

    def __init__(self, text, tokenizer=None):
        self.text = text
        self.tokenizer = tokenizer or CharTokenizer(text)
        self.ids = torch.tensor(self.tokenizer.encode(text))

    def __repr__(self):
        return (f"CharData: {len(self.text):,} characters, "
                f"vocabulary of {len(self.tokenizer)}")


def load_text(source="shakespeare", cache="tinyshakespeare.txt", tokenizer=None,
              characters=None):
    """Load a text corpus for character-level modelling.

    `source` may be "shakespeare", a URL, or a local file path.
    The Shakespeare corpus is cached locally after the first download.
    Pass `tokenizer=` to encode with an existing model's tokenizer
    instead of building a fresh character-level one, for fine-tuning.
    Pass `characters=` to keep only the first that many characters.
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
    if characters is not None:
        text = text[:characters]
    return CharData(text, tokenizer=tokenizer)


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

    def device(self):
        """Where the model's weights are, "cpu" or "cuda"."""
        return next(self.model.parameters()).device

    def n_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self):
        return (f"{self.name}: {self.n_parameters():,} parameters, "
                f"vocabulary of {len(self.tokenizer)}")


def load_lm(name="gpt2"):
    """Load a pretrained language model, e.g. "gpt2" or "distilgpt2"."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="eager",   # eager so attention is returned
        dtype=torch.float32)                 # full precision, as on CPU
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
    ids = torch.tensor([lm.tokenizer.encode(prompt)]).to(lm.device())
    with torch.no_grad():
        probs = F.softmax(lm.logits(ids)[0, -1], dim=-1).cpu()
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
    ids = torch.tensor([lm.tokenizer.encode(text)]).to(lm.device())
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
    ids = torch.tensor([lm.tokenizer.encode(prompt)]).to(lm.device())
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
    ids = torch.tensor([lm.tokenizer.encode(sentence)]).to(lm.device())
    with torch.no_grad():
        out = lm.model(ids, output_attentions=True)
    att = out.attentions[layer][0, head].cpu()
    tokens = [lm.tokenizer.decode([int(i)]) for i in ids[0]]
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
    ids = torch.tensor([lm.tokenizer.encode(sentence)]).to(lm.device())
    with torch.no_grad():
        out = lm.model(ids, output_attentions=True)
    tokens = [lm.tokenizer.decode([int(i)]) for i in ids[0]]
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 5))
    for ax, (layer, head) in zip(axes, heads):
        att = out.attentions[layer][0, head].cpu()
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
    return [(lm.tokenizer.decode([int(i)]), v.item())
            for v, i in zip(top.values, top.indices)]


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
            x, y = x.to(self.lm.device()), y.to(self.lm.device())
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


# ---------------------------------------------------------------------------
# A different job for a pretrained transformer: classification, no training
# ---------------------------------------------------------------------------

_classifier = None


def classify(text, labels):
    """Score how well each of `labels` fits `text`, with no training at all.

    Everything else in this module does next-token prediction, generation
    or fine-tuning included. This is a genuinely different use of a
    pretrained transformer: a model trained to judge whether one sentence
    follows from another is reused to judge whether a label follows from
    a piece of text. Scores always sum to one across the labels given,
    so the model can never say "none of these", only "which of these is
    most likely of the ones I was offered". The underlying model is
    loaded once and reused on later calls, since it is a large download.
    """
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline("zero-shot-classification",
                               model="facebook/bart-large-mnli")
    result = _classifier(text, labels)
    return list(zip(result["labels"], result["scores"]))


def show_classify(text, labels):
    """Print the text and how well each of `labels` fits it, best first."""
    print(text)
    for label, score in classify(text, labels):
        print(f"  {score:.3f}  {label}")


def set_seed(seed=0):
    """Fix the random number generator, so a run can be repeated exactly."""
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Text classification with GPT-style models
#
# One worked problem, solved several ways: prompting a pretrained model,
# reading features out of it, fine-tuning it, or training one from
# scratch. Every method returns its predictions for the test set, and
# `evaluate` scores them against the true labels and records the result,
# so `show_results` can put every method in one table.
#
# These functions also need the datasets and scikit-learn packages,
# imported only when used. They put models on the GPU when there is one.
# ---------------------------------------------------------------------------

DATASETS = {
    # name: (Hugging Face dataset, text column, split used as the test set)
    "emotion": ("dair-ai/emotion", "text", "test"),
    "sst2": ("stanfordnlp/sst2", "sentence", "validation"),
}

results = {}    # filled in by evaluate, shown by show_results


class TextData:
    """Labelled texts, split into training, validation and test sets.

    Each set is a list of texts and a matching array of integer labels.
    `label_names[i]` is the name of label i.
    """

    def __init__(self, name, label_names, train, validation, test):
        self.name = name
        self.label_names = label_names
        self.train_texts, self.train_labels = train
        self.val_texts, self.val_labels = validation
        self.test_texts, self.test_labels = test

    def __repr__(self):
        return (f"{self.name}: {len(self.train_texts)} training, "
                f"{len(self.val_texts)} validation, {len(self.test_texts)} test "
                f"texts, labels {self.label_names}")


def load_classification_data(name="emotion", train=2000, validation=500, test=500):
    """Load small random subsets of a labelled text dataset.

    "emotion": tweets labelled with one of six emotions (Saravia et al.,
    2018). "sst2": film-review sentences labelled positive or negative
    (Socher et al., 2013). SST-2's test labels are not public, so its
    official validation split is the test set here, and the validation
    set is taken from training texts not in the training subset.
    """
    import numpy as np
    from datasets import load_dataset
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of {list(DATASETS)}.")
    source, column, test_split = DATASETS[name]
    full = load_dataset(source)
    label_names = full["train"].features["label"].names

    def subset(split, start, size):
        rows = full[split].shuffle(seed=0).select(range(start, start + size))
        return list(rows[column]), np.array(rows["label"])

    if test_split == "validation":
        val = subset("train", train, validation)
    else:
        val = subset("validation", 0, validation)
    return TextData(name, label_names, subset("train", 0, train), val,
                    subset(test_split, 0, test))


def show_examples(data, n=5):
    """Print the first n training texts with their labels."""
    for text, label in zip(data.train_texts[:n], data.train_labels[:n]):
        print(f"{data.label_names[label]:>9}  {text}")


def show_counts(data, labels=None):
    """How many texts have each label, most common first.

    By default counts the true labels of the training set. Pass a
    method's predictions as `labels` to see what it predicts instead.
    """
    from collections import Counter
    if labels is None:
        labels = data.train_labels
    for label, count in Counter(labels).most_common():
        print(f"{data.label_names[label]:>9}  {count}")


def evaluate(data, predicted, name):
    """Print and record accuracy and macro F1 of predictions on the test set.

    Macro F1 is the average of the F1 scores of the separate classes, so
    a method that ignores the rare classes scores badly on it even when
    its accuracy looks respectable. If there are fewer predictions than
    test texts, they are compared against the first test texts only.
    """
    from sklearn.metrics import accuracy_score, f1_score
    true = data.test_labels[:len(predicted)]
    accuracy = accuracy_score(true, predicted)
    macro_f1 = f1_score(true, predicted, average="macro")
    results[name] = (accuracy, macro_f1, len(predicted))
    print(f"{name}: accuracy {accuracy:.3f}, macro F1 {macro_f1:.3f}")


def show_results():
    """Every method evaluated so far, in one table."""
    width = max(len(name) for name in results)
    print(f"{'':<{width}}  accuracy  macro F1  test texts")
    for name, (accuracy, macro_f1, n) in results.items():
        print(f"{name:<{width}}  {accuracy:8.3f}  {macro_f1:8.3f}  {n:10}")


# --- Baselines -------------------------------------------------------------

def predict_majority(data):
    """Predict the most common training label for every test text."""
    import numpy as np
    most_common = np.bincount(data.train_labels).argmax()
    return np.full(len(data.test_texts), most_common)


def predict_tfidf(data):
    """A bag of words: TF-IDF word counts, then logistic regression.

    Word order is ignored entirely. The regularisation strength C is
    tuned by 5-fold cross-validation on the training set only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    model = GridSearchCV(make_pipeline(TfidfVectorizer(),
                                       LogisticRegression(max_iter=2000)),
                         {"logisticregression__C": [1, 10, 100]},
                         cv=5, scoring="f1_macro")
    model.fit(data.train_texts, data.train_labels)
    print(f"chosen C: {model.best_params_['logisticregression__C']}")
    return model.predict(data.test_texts)


# --- Prompting: classification by next-token prediction --------------------

def _label_scores(lm, texts, template, labels):
    """Log-probability of each label as the next token after each prompt.

    This is plot_next_token again, read off for just the label words.
    Each label is represented by its first token, with a leading space,
    since that is how a word appears after "Emotion:" in running text.
    Returns a tensor of shape (number of texts, number of labels).
    """
    label_ids = [lm.tokenizer.encode(" " + label, add_special_tokens=False)[0]
                 for label in labels]
    if len(set(label_ids)) < len(labels):
        raise ValueError("Two labels begin with the same token. "
                         "Choose labels whose first words differ.")
    lm.model.to(DEVICE)
    scores = []
    for text in texts:
        ids = lm.tokenizer(template.format(text=text), return_tensors="pt").input_ids
        with torch.no_grad():
            next_token_logits = lm.logits(ids.to(DEVICE))[0, -1]
        scores.append(next_token_logits[label_ids].log_softmax(-1).cpu())
    return torch.stack(scores)


def show_prompt(data, template):
    """Print the prompt the model sees for the first test text."""
    print(template.format(text=data.test_texts[0]))


def predict_by_prompt(lm, data, template, n=None, calibrate=False):
    """Classify each test text by which label the model predicts next.

    The text is put into `template` where it says {text}, and the label
    with the highest next-token probability is the prediction. No
    training at all. `n` uses only the first n test texts, for slow
    models or long prompts.

    With calibrate=True, the model's label probabilities for prompts with
    no content at all ("N/A", "", "[MASK]") are taken as pure bias and
    subtracted from every score (Zhao et al., 2021, "Calibrate before use").
    """
    texts = data.test_texts[:n] if n else data.test_texts
    scores = _label_scores(lm, texts, template, data.label_names)
    if calibrate:
        bias = _label_scores(lm, ["N/A", "", "[MASK]"], template,
                             data.label_names).mean(dim=0)
        print("bias:", {name: round(p, 3) for name, p
                        in zip(data.label_names, bias.exp().tolist())})
        scores = scores - bias
    return scores.argmax(dim=1).numpy()


def compare_templates(lm, data, templates, n=200):
    """Validation accuracy of each prompt template. Returns the best one.

    Compare on the validation set, never the test set, so that the
    choice of prompt is not tuned to the test set.
    """
    import numpy as np
    accuracy = []
    for template in templates:
        scores = _label_scores(lm, data.val_texts[:n], template, data.label_names)
        accuracy.append((scores.argmax(dim=1).numpy() == data.val_labels[:n]).mean())
        print(f"{accuracy[-1]:.3f}  {template!r}")
    return templates[int(np.argmax(accuracy))]


def few_shot_template(data, template):
    """Put one labelled training example of each class before `template`.

    The model is still not trained, it is just shown examples in its
    context. The examples are the first of each class in the training set.
    """
    shots = ""
    for label, name in enumerate(data.label_names):
        i = list(data.train_labels).index(label)
        example = template.format(text=data.train_texts[i]) + f" {name}\n\n"
        shots += example.replace("{", "{{").replace("}", "}}")
    return shots + template


# --- Features: the language model's hidden states as inputs ----------------

def _embed(lm, texts, batch_size=32):
    """Mean hidden state of each text, at every layer.

    Returns an array of shape (layers + 1, number of texts, width).
    Index 0 is the token embeddings, before any attention.
    """
    if lm.tokenizer.pad_token is None:
        # GPT-2 has no padding token. Reusing the end-of-text token is the
        # standard workaround; the attention mask stops it mattering.
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    lm.model.to(DEVICE)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = lm.tokenizer(texts[start:start + batch_size], padding=True,
                             truncation=True, max_length=64,
                             return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hidden = lm.model(**batch, output_hidden_states=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1)    # ignore padding
        vectors.append(torch.stack([(h * mask).sum(1) / mask.sum(1)
                                    for h in hidden]).cpu())
    return torch.cat(vectors, dim=1).numpy()


def extract_features(lm, data):
    """Hidden-state features for every text in the training, validation
    and test sets: one vector per text, at every layer of the model.
    """
    features = {"train": _embed(lm, data.train_texts),
                "validation": _embed(lm, data.val_texts),
                "test": _embed(lm, data.test_texts)}
    layers, n, width = features["train"].shape
    print(f"{layers} layers (0 = token embeddings), {width} numbers per text")
    return features


def _probe():
    """A logistic regression classifier, on standardised features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def compare_layers(features, data):
    """Validation accuracy of a classifier on each layer's features.

    Prints and plots the accuracy layer by layer. Returns the best layer.
    """
    import numpy as np
    accuracy = []
    for layer in range(len(features["train"])):
        probe = _probe().fit(features["train"][layer], data.train_labels)
        accuracy.append(probe.score(features["validation"][layer], data.val_labels))
        print(f"layer {layer:>2}  validation accuracy {accuracy[-1]:.3f}")
    plt.figure()
    plt.plot(accuracy, marker="o")
    plt.xlabel("layer (0 = token embeddings)")
    plt.ylabel("validation accuracy")
    plt.title(f"{data.name}: read off each layer")
    plt.tight_layout()
    plt.show()
    return int(np.argmax(accuracy))


def predict_from_features(features, data, layer):
    """Train a classifier on one layer's features and predict the test set."""
    probe = _probe().fit(features["train"][layer], data.train_labels)
    return probe.predict(features["test"][layer])


# --- Fine-tuning, and training from scratch --------------------------------

def build_classifier(name, data):
    """A pretrained model with its next-token output layer replaced.

    The new output layer has one output per label, reads the hidden state
    of the final token, the only one that has attended to the whole text,
    and starts with random weights. e.g. build_classifier("distilgpt2", data)
    """
    from transformers import AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        name, id2label=dict(enumerate(data.label_names)))
    model.config.pad_token_id = tokenizer.pad_token_id
    return LanguageModel(model.to(DEVICE), tokenizer, context=64,
                         name=f"{name} classifier")


def build_tiny_classifier(data, layers=2, width=128, heads=2):
    """A small GPT classifier with random weights, using GPT-2's tokenizer.

    Same architecture family as build_classifier, but nothing pretrained:
    everything it knows about language it must learn from the training set.
    """
    from transformers import GPT2ForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=64,
                        n_embd=width, n_layer=layers, n_head=heads,
                        id2label=dict(enumerate(data.label_names)),
                        pad_token_id=tokenizer.pad_token_id)
    model = GPT2ForSequenceClassification(config)
    return LanguageModel(model.to(DEVICE), tokenizer, context=64,
                         name="tiny GPT classifier")


def _predict(classifier, texts, batch_size=64):
    """The classifier's predicted label for each text."""
    classifier.model.eval()
    predicted = []
    for start in range(0, len(texts), batch_size):
        batch = classifier.tokenizer(texts[start:start + batch_size], padding=True,
                                     truncation=True, max_length=64,
                                     return_tensors="pt").to(classifier.device())
        with torch.no_grad():
            predicted.append(classifier.model(**batch).logits.argmax(dim=-1).cpu())
    return torch.cat(predicted).numpy()


def predict_classifier(classifier, data):
    """The classifier's predicted label for each test text."""
    return _predict(classifier, data.test_texts)


def fit_classifier(classifier, data, epochs=2, learning_rate=5e-5, batch_size=16):
    """Train on the training set, reporting validation accuracy per epoch.

    The same training loop as every classifier in Weeks 1 and 2:
    shuffle, take a batch, compute the cross-entropy loss, step.
    """
    model, device = classifier.model, classifier.device()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(data.train_texts)).tolist()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = classifier.tokenizer([data.train_texts[i] for i in rows],
                                         padding=True, truncation=True,
                                         max_length=64, return_tensors="pt")
            labels = torch.tensor(data.train_labels[rows], device=device)
            loss = model(**batch.to(device), labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_accuracy = (_predict(classifier, data.val_texts) == data.val_labels).mean()
        print(f"epoch {epoch}   last batch loss {loss.item():.3f}   "
              f"validation accuracy {val_accuracy:.3f}")
    model.eval()
    return classifier


# --- Looking at the errors -------------------------------------------------

def plot_confusion(data, predicted, title=""):
    """Confusion matrix: true labels down the side, predictions across."""
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        data.test_labels[:len(predicted)], predicted,
        labels=range(len(data.label_names)), display_labels=data.label_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_errors(data, predicted, n=15):
    """Print the first n test texts the predictions got wrong."""
    errors = [(text, true, guess) for text, true, guess
              in zip(data.test_texts, data.test_labels, predicted) if true != guess]
    print(f"{len(errors)} errors out of {len(predicted)}")
    for text, true, guess in errors[:n]:
        print(f"true {data.label_names[true]:>8}   "
              f"predicted {data.label_names[guess]:>8}   {text}")
