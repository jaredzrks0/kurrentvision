"""
Cleans raw_book_texts.txt and outputs:
  - unique words (one per line)
  - n text segments of random word length (3–10), copied verbatim from the source
"""

import re
import random
import argparse
import math
from pathlib import Path


def clean_text(raw: str) -> str:
    # Remove lines that are purely metadata/headers (e.g. "Burgwall, den 10. Juni 18--.")
    # Collapse multiple spaces/newlines into single spaces
    text = raw

    # Remove control characters except newlines
    text = re.sub(r'[^\S\n]+', ' ', text)          # multiple spaces → single space
    text = re.sub(r'\n+', '\n', text)               # multiple newlines → single newline
    text = re.sub(r'[ \t]+\n', '\n', text)          # trailing spaces before newline
    text = re.sub(r'\n[ \t]+', '\n', text)          # leading spaces after newline

    # Remove arrows, fancy quotes, dashes, brackets, underscores, double hyphens
    text = re.sub(r'--+', '', text)
    text = re.sub(r'[»«""–—\[\]_]', '', text)
    # Remove any remaining non-German characters
    text = re.sub(r'[^\w\s\.,;:!?\-\'\"()]', '', text, flags=re.UNICODE)
    text = text.replace('_', '')

    # Collapse any remaining multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()

def extract_unique_words(tokens: list[str]) -> list[str]:
    stripped = [tok.strip('.,;:!?\'\"()-') for tok in tokens]
    counts: dict[str, int] = {}
    for w in stripped:
        if w:
            counts[w] = counts.get(w, 0) + 1
    result = []
    for word, count in counts.items():
        n = max(1, math.floor(math.log2(count)))
        result.extend([word] * n)
    return result


def sample_segments(tokens: list[str], n: int) -> list[str]:
    """Sample n segments of random length 3–10 words from the token list."""
    segments = []
    max_start = len(tokens) - 3  # need at least 3 words
    for _ in range(n):
        length = random.randint(3, 10)
        start = random.randint(0, max(0, len(tokens) - length))
        segment = ' '.join(tokens[start:start + length])
        segments.append(segment)
    return segments


def main():
    parser = argparse.ArgumentParser(description="Prepare text data for synthetic image generation")
    parser.add_argument('--input', type=str, default='data/raw_book_texts.txt')
    parser.add_argument('--output', type=str, default='data/synthetic_texts.txt')
    parser.add_argument('--segments', type=int, default=1000, help='Number of random segments to generate')
    parser.add_argument('--seed', type=int, default=6500)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = input_path.read_text(encoding='utf-8')
    cleaned = clean_text(raw)
    
    # Tokenize to word level
    tokens = cleaned.split()

    unique_words = extract_unique_words(tokens)
    segments = sample_segments(tokens, args.segments)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(word + '\n')
        for seg in segments:
            f.write(seg + '\n')

    print(f"Unique words: {len(unique_words)}")
    print(f"Segments: {len(segments)}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
