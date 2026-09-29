"""TMDB Movie Overview Text Preprocessing Pipeline.

This script executes an end-to-end NLP preprocessing workflow on TMDB movie overviews:
- Noise removal (HTML, URLs, punctuation, emojis, emoticons)
- Stopword, frequent-word, and rare-word removal
- Chat-word slang expansion and whitespace normalization
- Word count statistics
- Stemming (Porter) and Lemmatization (WordNet)
- Universal Part-of-Speech (POS) tagging
"""

import argparse
import os
import re
import string
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd


def ensure_nltk_resources() -> None:
    """Download required NLTK resources quietly if missing."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "universal_tagset",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            print(f"Warning: Failed downloading NLTK resource '{resource}': {exc}")


def remove_html(text: str) -> str:
    """Strip HTML tags using BeautifulSoup."""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")


def remove_urls(text: str) -> str:
    """Remove web URLs from text."""
    return re.sub(r"http\S+|www\S+|https\S+", "", str(text))


def remove_punctuation(text: str) -> str:
    """Remove standard ASCII punctuation marks."""
    return str(text).translate(str.maketrans("", "", string.punctuation))


def compile_emoji_regex() -> re.Pattern:
    """Compile Unicode ranges matching emojis."""
    return re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )


def compile_emoticon_regex() -> str:
    """Pattern string matching common emoticons."""
    return r"""
        [:;=8][\-^']?[)(DPp]
        |[:;=8][\-^']?[(/\\]
    """


CHAT_WORDS = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "btw": "by the way",
    "idk": "i do not know",
    "omg": "oh my god",
    "thx": "thanks",
    "pls": "please",
}


def pos_tag_text(text: str) -> list[tuple[str, str]]:
    """Tokenize and tag text with universal POS tags."""
    tokens = nltk.word_tokenize(str(text))
    return nltk.pos_tag(tokens, tagset="universal")


def run_pipeline(input_path: Path, output_path: Path, top_frequent: int = 10) -> pd.DataFrame:
    """Execute the full text preprocessing pipeline."""
    ensure_nltk_resources()

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Initial records: {initial_count} rows, {df.shape[1]} columns")

    # 1. Missing value handling
    df = df.dropna(subset=["Overview"]).copy()
    print(f"Records after dropping missing 'Overview': {len(df)} (dropped {initial_count - len(df)})")

    # 2. Lowercasing
    df["clean_overview"] = df["Overview"].astype(str).str.lower()

    # 3. Remove HTML tags
    df["clean_overview"] = df["clean_overview"].apply(remove_html)

    # 4. Remove URLs
    df["clean_overview"] = df["clean_overview"].apply(remove_urls)

    # 5. Remove punctuation
    df["clean_overview"] = df["clean_overview"].apply(remove_punctuation)

    # 6. Remove emojis
    emoji_regex = compile_emoji_regex()
    df["clean_overview"] = df["clean_overview"].str.replace(emoji_regex, "", regex=True)

    # 7. Remove emoticons
    emoticon_regex = compile_emoticon_regex()
    df["clean_overview"] = df["clean_overview"].apply(
        lambda t: re.sub(emoticon_regex, "", t, flags=re.VERBOSE)
    )

    # 8. Remove stopwords
    stop_words = set(stopwords.words("english"))
    df["clean_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(w for w in t.split() if w not in stop_words)
    )

    # 9. Remove top frequent words
    all_words = " ".join(df["clean_overview"]).split()
    word_freq = Counter(all_words)
    frequent_words = {w for w, _ in word_freq.most_common(top_frequent)}
    print(f"Removing top {top_frequent} frequent words: {sorted(frequent_words)}")
    df["clean_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(w for w in t.split() if w not in frequent_words)
    )

    # 10. Remove rare words (frequency == 1)
    all_words_after_freq = " ".join(df["clean_overview"]).split()
    word_freq_after_freq = Counter(all_words_after_freq)
    rare_words = {w for w, c in word_freq_after_freq.items() if c == 1}
    print(f"Removing {len(rare_words)} rare words (hapax legomena)")
    df["clean_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(w for w in t.split() if w not in rare_words)
    )

    # 11. Chat word expansion
    df["clean_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(CHAT_WORDS.get(w, w) for w in t.split())
    )

    # 12. Whitespace normalization
    df["clean_overview"] = df["clean_overview"].apply(lambda t: re.sub(r"\s+", " ", t).strip())

    # 13. Word count comparison
    df["original_word_count"] = df["Overview"].astype(str).str.split().str.len()
    df["clean_word_count"] = df["clean_overview"].str.split().str.len()
    avg_orig = df["original_word_count"].mean()
    avg_clean = df["clean_word_count"].mean()
    print(f"Average original word count: {avg_orig:.2f}")
    print(f"Average cleaned word count:  {avg_clean:.2f} ({((avg_orig - avg_clean) / avg_orig) * 100:.1f}% reduction)")

    # 14. Stemming
    stemmer = PorterStemmer()
    df["stemmed_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(stemmer.stem(w) for w in t.split())
    )

    # 15. Lemmatization
    lemmatizer = WordNetLemmatizer()
    df["lemmatized_overview"] = df["clean_overview"].apply(
        lambda t: " ".join(lemmatizer.lemmatize(w) for w in t.split())
    )

    # 16. POS tagging
    print("Computing Universal POS tags...")
    df["pos_tags"] = df["clean_overview"].apply(pos_tag_text)

    # 17. Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path} ({len(df)} rows, {df.shape[1]} columns)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="TMDB Movie Overview Text Preprocessing Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tmdb_movies.csv"),
        help="Path to raw TMDB movies CSV file (default: data/tmdb_movies.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tmdb_preprocessed.csv"),
        help="Path to output preprocessed CSV file (default: data/tmdb_preprocessed.csv)",
    )
    parser.add_argument(
        "--top-frequent",
        type=int,
        default=10,
        help="Number of most frequent terms to prune (default: 10)",
    )
    args = parser.parse_args()

    # Fallback to local file if running from another directory
    input_file = args.input
    if not input_file.exists():
        fallback_candidates = [
            Path("../data/tmdb_movies.csv"),
            Path("tmdb_movies.csv"),
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                input_file = candidate
                break

    output_file = args.output
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_path=input_file, output_path=output_file, top_frequent=args.top_frequent)


if __name__ == "__main__":
    main()
