# nettlelang

Minimal utilities for demonstrating language models, a companion to nettle but entirely independent of it.

## Install

    pip install git+https://github.com/mark-andrews/nettlelang.git

To pick up later updates, re-run the same command with `--force-reinstall --no-deps`:

    pip install --force-reinstall --no-deps git+https://github.com/mark-andrews/nettlelang.git

The `transformers` dependency is deliberately not declared in the package metadata, so install it explicitly:

    pip install transformers

## Example

    import nettlelang as nl

    gpt2 = nl.load_lm("gpt2")
    nl.show_tokens(gpt2, "Unbelievable happenings in Nottingham")
    nl.plot_next_token(gpt2, "The cat sat on the")
    print(nl.generate(gpt2, "Once upon a time", temperature=0.8))
    nl.plot_attention(gpt2, "The dog chased the ball because it was tired")
