import os
from typing import BinaryIO

import regex as re
import multiprocessing as mp


from collections import Counter
from collections import defaultdict

import time
import json
import tracemalloc

import cProfile
import pickle



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    split_special_token = [s.encode("utf-8") for s in split_special_token]
        # assert isinstance(s, bytes), "Must represent special token as a bytestring"


    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_positions = [
                mini_chunk.find(tok)
                for tok in split_special_token
                if mini_chunk.find(tok) != -1
            ]

            if found_positions:
                found_at = min(found_positions)
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(chunk, special_tokens):
    counter = defaultdict(int)
    split_delimiter = "|".join(re.escape(s) for s in special_tokens)
    splitted_chunks = re.split(split_delimiter, chunk)

    for part in splitted_chunks:
        if not part:
            continue
        for m in re.finditer(PAT, part):
            counter[m.group(0).encode('utf-8')] += 1
    return counter

def _pre_tokenize_worker(args):
    chunk, special_tokens = args
    return pre_tokenize(chunk, special_tokens)

def _pairs_worker(items):
    pairs_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for w_bt, freq in items:
        for i in range(len(w_bt) - 1):
            pair = (w_bt[i], w_bt[i + 1])
            pairs_counts[pair] += freq #TUPLE
            pair_to_words[pair].add(w_bt)

    return pairs_counts, pair_to_words

def get_initial_pairs(words_as_bytes):
    pairs_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for w_bt, freq in words_as_bytes.items():
        for i in range(len(w_bt) - 1):
            pair = (w_bt[i], w_bt[i + 1])
            pairs_counts[pair] += freq #TUPLE
            pair_to_words[pair].add(w_bt)

    return pairs_counts, pair_to_words

def merge(words_as_bytes, best_pair, pairs, pair_to_words):

    # freq_bp = pairs[best_pair]
    l, r = best_pair
    merged_pair = l+r
    changed_words = {}
    
    affected_words = list(pair_to_words[best_pair])
    pair_to_words[best_pair].clear()

    for w_bt in affected_words:
        freq = words_as_bytes[w_bt]
        # scan this word only
        idxs = []
        for i in range(len(w_bt) - 1):
            if w_bt[i] == l and w_bt[i + 1] == r:
                idxs.append(i)

        if not idxs:
            continue

        # rebuild word with merges
        merged = []
        i = 0
        j = 0
        while i < len(w_bt):
            if j < len(idxs) and idxs[j] == i:
                merged.append(merged_pair)
                i += 2
                j += 1
                while j < len(idxs) and idxs[j] < i:
                    j += 1
            else:
                merged.append(w_bt[i])
                i += 1
        new_w = tuple(merged)

        # update pair counts & pair_to_words
        # remove old pairs from this word
        for i in range(len(w_bt) - 1):
            pair = (w_bt[i], w_bt[i+1])
            pairs[pair] -= freq
            pair_to_words[pair].discard(w_bt)

        # add new pairs from new word
        for i in range(len(new_w) - 1):
            pair = (new_w[i], new_w[i+1])
            pairs[pair] += freq
            pair_to_words[pair].add(new_w)

        changed_words[w_bt] = new_w

    if changed_words:
        for old_w, new_w in changed_words.items():
            freq = words_as_bytes.pop(old_w)
            words_as_bytes[new_w] = words_as_bytes.get(new_w, 0) + freq

    pairs[best_pair] -= pairs[best_pair]



def tokenizer_merge(frequency_table, vocab_size, vocab):
    words_as_bytes = {}
    for word, freq in frequency_table.items():
        # b = word.encode("utf-8")
        seq_bytes = tuple(bytes([x]) for x in word)
        if len(seq_bytes) > 1:
            words_as_bytes[seq_bytes] = freq
            # print(seq_bytes)

    # remove leading space?
    pairs = defaultdict(int)
    pair_to_words = defaultdict(set)
    items = list(words_as_bytes.items())
    if items:
        
        import sys, multiprocessing as mp
        mp.set_executable(sys.executable)

        ctx = mp.get_context("spawn")
        num_workers = min(4, mp.cpu_count())
        if len(items) < 1000 or num_workers == 1:
            pairs, pair_to_words = get_initial_pairs(words_as_bytes)
        else:
            chunk_size = max(1, len(items) // num_workers)
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            with ctx.Pool(processes=num_workers) as pool:
                for part, part_map in pool.map(_pairs_worker, chunks, chunksize=1):
                    for k, v in part.items():
                        pairs[k] += v
                    for k, wordset in part_map.items():
                        pair_to_words[k].update(wordset)

    merges = []
    while len(vocab) < vocab_size:
        best_pair = max( pairs.items(), key=lambda x: (x[1], x[0]))[0] #remove ties lexicografically
        merge(words_as_bytes, best_pair, pairs, pair_to_words)
        l, r = best_pair
        vocab[len(vocab)] = l+r
        merges.append((l, r)) #TUPLE

    return vocab, merges

import sys
import multiprocessing as mp

def train_bpe(input_path, vocab_size, special_tokens):
    # mp.set_executable(sys.executable)  # must be set before any Pool/Process
    num_processes = 4
    vocab = {}
    next_id = 0

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id +=1

    if next_id >= vocab_size:
        print("vocab hit limit before tokenizing")

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, special_tokens)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunks.append(file.read(end - start).decode("utf-8", errors="ignore"))

    ctx = mp.get_context("spawn")
    num_workers = min(num_processes, mp.cpu_count())
    with ctx.Pool(processes=num_workers) as pool:
        pretokenized = list(pool.imap(_pre_tokenize_worker, ((c, special_tokens) for c in chunks), chunksize=1))

    # with open(input_path, "rb") as file:
    #     file.seek(0)
    #     corpus = file.read().decode("utf-8", errors="ignore")
    #     pretokenized = pre_tokenize(corpus, special_tokens)

    frequency_table = Counter()
    for ft in pretokenized:
        frequency_table.update(ft)

    vocab, merges = tokenizer_merge(frequency_table, vocab_size, vocab)

    return vocab, merges


import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a model with configurable dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to train tokenizer on")
    args = parser.parse_args()

    if args.dataset == "smaller":
        print("here")
        input_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/smaller_test.txt"
        vocab_size = 263
        vocab_path = "bpe_vocab_smaller.pkl"
        merges_path = "bpe_merges_smaller.pkl"
    else:
        print("there")
        input_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-train.txt"
        vocab_size = 10000
        vocab_path = "bpe_vocab.pkl"
        merges_path = "bpe_merges.pkl"

    special_tokens = ["<|endoftext|>"]

    # start = time.perf_counter()
    # pr = cProfile.Profile()
    # pr.enable()

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # pr.disable()
    # pr.print_stats(sort="cumtime")
    # elapsed = time.perf_counter() - start
    # print(f"time: {elapsed:.2f}s")


    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # token_id, token_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    # print("max token id:", token_id)
    # print("max token length:", len(token_bytes))
    # print("token bytes:", token_bytes)








if __name__ == '__main__':
    main()

    # input_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000

    # # input_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/smaller_test.txt"
    # # vocab_size = 263
    # special_tokens = ["<|endoftext|>"]


    # start = time.perf_counter()
    # pr = cProfile.Profile()
    # pr.enable()


    # vocab, merges = train_bpe(input_path, vocab_size, special_tokens)


    # pr.disable()
    # pr.print_stats(sort="cumtime")
    # elapsed = time.perf_counter() - start
    # print(f"time: {elapsed:.2f}s")

    # with open("bpe_vocab_smaller.pkl", "wb") as f:
    #     pickle.dump(vocab, f)

    # with open("bpe_merges_smaller.pkl", "wb") as f:
    #     pickle.dump(merges, f)

    # token_id, token_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    # print("max token id:", token_id)
    # print("max token length:", len(token_bytes))
    # print("token bytes:", token_bytes)


    # # vocab_out = {int(k): list(v) for k, v in vocab.items()}
    # # with open("bpe_vocab_smaller.json", "w", encoding="utf-8") as f:
    # #     json.dump(vocab_out, f)

    # # with open("bpe_merges_smaller.txt", "wb", encoding="utf-8") as f:
    # #     for a, b in merges:
    # #         f.write(f"{a} {b}\n")

    # # with open("bpe_merges_smaller.txt", "w", encoding="utf-8") as f:
    # #     for a, b in merges:
    # #         f.write(f"{a.hex()} {b.hex()}\n")

