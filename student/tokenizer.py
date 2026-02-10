import json
import regex as re
import pickle
import multiprocessing as mp
from tqdm import tqdm
# from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _find_chunk_boudaries(text, desired_num_chunks, split_special_token):
    # split_special_token = list(split_special_token)

    file_size =len(text)
    chunk_size = file_size // desired_num_chunks

    if not split_special_token:
        print("NO CHUNKS")
        return [0, len(text)]


    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        # file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = text[initial_position:initial_position+mini_chunk_size]  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == "":
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
    print(sorted(set(chunk_boundaries)))
    return sorted(set(chunk_boundaries))


def _merge_word_worker(args):
    word, merges, special_tokens, cache = args

    if special_tokens and word.decode("utf-8") in special_tokens:
        return [word], cache
    
    if cache and word in cache:
        return cache[word], cache
        
    seq_bytes = [bytes([x]) for x in word]
    for l, r in merges:
        merged = l + r
        new_word = []
        i = 0
        n = len(seq_bytes)
        while i < n:
            if i < n - 1 and seq_bytes[i] == l and seq_bytes[i + 1] == r:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(seq_bytes[i])
                i += 1
        seq_bytes = new_word

    cache[word] = seq_bytes
    return seq_bytes, cache


def _get_ids_worker(args):
    parts, rev_vocab = args
    return [rev_vocab.get(part) for part in parts]

def _pre_tokenize_worker(args):
    chunk, special_tokens = args
    pre_tokenized = []

    if special_tokens:
        split_delimiter = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
        splitted_chunks = re.split(split_delimiter, chunk)
        for part in splitted_chunks:
            if not part:
                continue
            if part in special_tokens:
                pre_tokenized.append(part.encode("utf-8"))  # keep whole token
                continue
            for m in re.finditer(PAT, part):
                pre_tokenized.append(m.group(0).encode("utf-8"))

    else:   
        for m in re.finditer(PAT, chunk):
            pre_tokenized.append(m.group(0).encode("utf-8"))


    return pre_tokenized

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Handle special tokens not in vocab
        self.rev_vocab = {bytes(v): k for k, v in self.vocab.items()}
        if self.special_tokens:
            for special_tk in self.special_tokens:
                sp_t = special_tk.encode("utf-8")
                if sp_t not in self.rev_vocab:
                    self.vocab[len(self.vocab)] = sp_t

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text):

        num_workers = min(4, mp.cpu_count())
        print(f"Num workers: {num_workers}")

        boundaries = _find_chunk_boudaries(text, num_workers, self.special_tokens)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunks.append(text[start:end])

        ctx = mp.get_context("spawn")
        pre_tokenized = []
        with ctx.Pool(processes=num_workers) as pool:
            it = pool.imap(_pre_tokenize_worker, ((c, self.special_tokens) for c in chunks), chunksize=1)
            for part in tqdm(it, total=len(chunks), desc="Pre-tokenize", unit="chunk"):
                pre_tokenized.extend(part)

        # pre_tokenized = self._pre_tokenize(text)

        # if len(pre_tokenized) < 1000 or num_workers == 1:


        # merged_text = []
        # cache = {}
        # for word in tqdm(pre_tokenized, desc="Merging", unit="word"):
        #     merged, cache = _merge_word_worker((word, self.merges, self.special_tokens, cache))
        #     merged_text.append(merged)

        merged_text = self._merge(pre_tokenized)
        
        # encoded_tokens = self._get_ids(merged_text)
        # return encoded_tokens

        # with ctx.Pool(processes=num_workers) as pool:
        #     cache = {}
        #     merge_iter = pool.imap(
        #         _merge_word_worker,
        #         ((word, self.merges, self.special_tokens, cache) for word in pre_tokenized),
        #         chunksize=chunksize,
        #     )
        #     merged_text = list(tqdm(merge_iter, total=len(pre_tokenized), desc="Merging", unit="word"))

        # ctx = mp.get_context("spawn")
        # chunksize = max(10, len(pre_tokenized) // (num_workers * 10))
        # print(f"Chunk size: {chunksize}")

        # with ctx.Pool(processes=num_workers) as pool:
        #     ids_iter = pool.imap(
        #         _get_ids_worker,
        #         ((parts, self.rev_vocab) for parts in merged_text),
        #         chunksize=chunksize,
        #     )
        #     token_lists = list(tqdm(ids_iter, total=len(merged_text), desc="Mapping IDs", unit="word"))

        token_list = list(self._get_ids(merged_text))
        return token_list
    
        # encoded_tokens = []
        # for lst in token_list:
        #     encoded_tokens.extend(lst)
        # return encoded_tokens

    def _pre_tokenize(self, chunk):
        pre_tokenized = []

        if self.special_tokens:
            split_delimiter = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            splitted_chunks = re.split(split_delimiter, chunk)
            for part in splitted_chunks:
                if not part:
                    continue
                if part in self.special_tokens:
                    pre_tokenized.append(part.encode("utf-8"))  # keep whole token
                    continue
                for m in re.finditer(PAT, part):
                    pre_tokenized.append(m.group(0).encode("utf-8"))

        else:   
            for m in re.finditer(PAT, chunk):
                pre_tokenized.append(m.group(0).encode("utf-8"))


        return pre_tokenized
    
    def _get_ids(self, merged_text):
        token_ids = []
        for word in merged_text:
            for part in word:
                key = self.rev_vocab.get(part)
                token_ids.append(key)
        return token_ids

    def _merge(self, pre_tokenized_text):
        merged_text = []
        cache = {}
        for word in tqdm(pre_tokenized_text, desc="Merging", unit="word"):
            key = word if isinstance(word, (bytes, bytearray)) else tuple(word)
            if key in cache:
                merged_text.append(cache[key])
                continue
            if self.special_tokens and isinstance(word, (bytes, bytearray)) and word.decode("utf-8") in self.special_tokens:
                merged = [bytes(word)]
            else:
                seq_bytes = [bytes([x]) for x in word]
                for l, r in self.merges:
                    new_word = []
                    i=0
                    n = len(seq_bytes)
                    while i<n:
                        if i < n-1 and seq_bytes[i] == l and seq_bytes[i+1] == r:
                            new_word.append(l+r)
                            i += 2
                        else:
                            new_word.append(seq_bytes[i])
                            i += 1

                    seq_bytes = new_word
                merged = seq_bytes
            cache[key] = merged
            merged_text.append(merged)

        return merged_text

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for tok_id in self.encode(chunk):
                yield tok_id

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")



import argparse
import time
import numpy as np
import cProfile

def main():

    ###### DEBUG AREA - START
    # dataset_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-valid.txt"
    # vocab_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_vocab.pkl"
    # merges_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_merges.pkl"

    # tk = Tokenizer.from_files(vocab_path, merges_path)
    # with open(dataset_path, "r") as file:
    #         text = file.read()

    # start = time.perf_counter()
    # out1 = tk.encode(text)
    # elapsed = time.perf_counter() - start
    # print(f"time: {elapsed:.2f}s")

    ###### DEBUG AREA - END


    parser = argparse.ArgumentParser(description="Encode and decode text using already trained tokenizer")
    parser.add_argument("--corpus", type=str, help="Text to encode and decode")
    parser.add_argument("--n_documents", type=int, help="Number of documents from the corpus to encode")
    args = parser.parse_args()

    if args.corpus == "smaller":
        dataset_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/smaller_test.txt"
        vocab_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_vocab_smaller.pkl"
        merges_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_merges_smaller.pkl"

        with open(dataset_path, "r") as file:
            text = file.read()

        tk = Tokenizer.from_files(vocab_path, merges_path)
        out1 = tk.encode(text)

    else:
        dataset_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-valid.txt"
        vocab_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_vocab.pkl"
        merges_path = "/Users/juliagontijolopes/Desktop/LLMBuilder/Assignment1/nyu-llm-reasoners-a1/bpe_merges.pkl"

        if args.n_documents:
            print("chunks")
            with open(dataset_path, "r", encoding="utf-8") as file:
                docs = []
                curr = []
                for line in file:
                    if "<|endoftext|>" in line:
                        parts = line.split("<|endoftext|>")
                        # add text before token
                        curr.append(parts[0])
                        docs.append("".join(curr))
                        if len(docs) == args.n_documents:
                            break
                        # start next doc with remainder (if any)
                        curr = [parts[1]] if len(parts) > 1 else []
                    else:
                        curr.append(line)

            pr = cProfile.Profile()
            pr.enable()
            for idx, doc in enumerate(docs):
                text = doc + "<|endoftext|>"
                tk = Tokenizer.from_files(vocab_path, merges_path)
                start = time.perf_counter()
                # pr = cProfile.Profile()
                # pr.enable()
                out1 = tk.encode(text)
                # pr.disable()
                # pr.print_stats(sort="cumtime")
                elapsed = time.perf_counter() - start
                bts = len(text.encode("utf-8"))
                tks = len(out1)
                print(f"Document {idx}\nNum bytes: {bts}\nNum tokens: {tks}\n### Compression ratio: bytes/tokens = {bts/tks}")
                print(f"Time: {elapsed:.2f}s\n### Bytes per second:{bts/elapsed:.2f}\n\n")
            pr.disable()
            # pr.print_stats(sort="cumtime")

        else:
            print("full")
            with open(dataset_path, "r", encoding="utf-8") as file:
                text = file.read()

            special_tokens = ["<|endoftext|>"]
            
            tk = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
            start = time.perf_counter()
            pr = cProfile.Profile()
            pr.enable()
            out1 = tk.encode(text)
            pr.disable()
            pr.print_stats(sort="cumtime")
            elapsed = time.perf_counter() - start
            print(f"time: {elapsed:.2f}s")

            arr = np.array(out1, dtype=np.uint16)
            np.save("TinyStoriesV2-GPT4-train-caching-attempt.npy", arr)



if __name__ == '__main__':
    main()
