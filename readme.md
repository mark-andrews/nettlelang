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

## Text classification

Version 0.0.2 adds functions for classifying text with GPT-style models.
They also need the `datasets` and `scikit-learn` packages, which are imported only when these functions are used, and they use a GPU automatically when there is one.
Every method returns its predictions for the test set, and `evaluate` scores and records them.

    emotion = nl.load_classification_data("emotion")
    nl.evaluate(emotion, nl.predict_tfidf(emotion), "TF-IDF baseline")

    gpt2 = nl.load_lm("gpt2")
    predicted = nl.predict_by_prompt(gpt2, emotion, "Text: {text}\nEmotion:")
    nl.evaluate(emotion, predicted, "GPT-2 zero-shot")

    classifier = nl.build_classifier("distilgpt2", emotion)
    nl.fit_classifier(classifier, emotion, epochs=2)
    nl.evaluate(emotion, nl.predict_classifier(classifier, emotion), "fine-tuned")

    nl.show_results()
