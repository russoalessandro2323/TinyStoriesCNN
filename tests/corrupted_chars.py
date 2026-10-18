from datasets import load_dataset

def test_corrupted_chars():
    # 1. Load the dataset split
    print("Loading TinyStories split...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # 2. Define the corrupted sequences and artifacts to search for
    patterns = ["â€œ", "â€", "â€™", "â€˜", "â€”", "â€“", "``"]
    counts = {p: 0 for p in patterns}
    samples = {p: [] for p in patterns}

    # 3. Scan the dataset rows
    print("Scanning dataset for corrupted character patterns...\n")
    for row in dataset:
        text = row["text"]
        for pattern in patterns:
            if pattern in text:
                counts[pattern] += 1
                # Save up to 2 sample snippets per pattern
                if len(samples[pattern]) < 2:
                    # Find substring location and slice context around it
                    idx = text.find(pattern)
                    start = max(0, idx - 20)
                    end = min(len(text), idx + 30)
                    snippet = text[start:end].replace("\n", " ")
                    samples[pattern].append(snippet)

    # 4. Display results
    print("=== Pattern Occurrences ===")
    for pattern, count in counts.items():
        print(f"Pattern '{pattern}': found in {count:,} rows")

    print("\n=== Snippet Samples ===")
    for pattern, snippets in samples.items():
        print(f"\nTarget pattern: '{pattern}'")
        for s in snippets:
            print(f"  --> ...{s}...")


"""

OUTPUT: 

Loading TinyStories split...
Scanning dataset for corrupted character patterns...

=== Pattern Occurrences ===
Pattern 'â€œ': found in 92,178 rows
Pattern 'â€': found in 129,567 rows
Pattern 'â€™': found in 86,222 rows
Pattern 'â€˜': found in 1,711 rows
Pattern 'â€”': found in 3,917 rows
Pattern 'â€“': found in 9,235 rows
Pattern '``': found in 51 rows

=== Snippet Samples ===

Target pattern: 'â€œ'
  --> ...at her from behind, â€œBe careful not to touch it!...
  --> ...her with a friendly â€œhiâ€ and the deer said â€œh...

Target pattern: 'â€'
  --> ...at her from behind, â€œBe careful not to touch it!...
  --> .... She took her daddyâ€™s tie and put it in the dir...

Target pattern: 'â€™'
  --> ...and replied, â€œThatâ€™s very wise young lady. Now...
  --> .... She took her daddyâ€™s tie and put it in the dir...
...

Target pattern: '``'
  --> ...ck smiled and said, ``you deserve it. Keep being n...
  --> ...  The brother said, ``I'm looking for my toe! I ca...

"""