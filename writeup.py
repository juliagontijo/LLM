# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf8_bytes_to_str_wrong("é".encode("utf-8")))



# # ORIGINAL IMPLEMENTATION OF BPE - FROM ARTICLE
# import re, collections
# def get_stats(vocab):
#     pairs = collections.defaultdict(int)
#     for word, freq in vocab.items():
#         symbols = word.split()
#         for i in range(len(symbols)-1):
#             pairs[symbols[i],symbols[i+1]] += freq
#     return pairs

# def merge_vocab(pair, v_in):
#     v_out = {}
#     bigram = re.escape(' '.join(pair))
#     p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#     for word in v_in:
#         w_out = p.sub(''.join(pair), word)
#         v_out[w_out] = v_in[word]
#     return v_out

# vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
# 'n e w e s t </w>':6, 'w i d e s t </w>':3}
# num_merges = 10

# for i in range(num_merges):
#     pairs = get_stats(vocab)
#     # print(pairs)
#     best = max(pairs, key=pairs.get)
#     vocab = merge_vocab(best, vocab)
#     print(vocab)
#     print(best)


# GETTING INTUITION FOR BPE TOKENIZATION EX 2

import json
import heapq
with open("bpe_vocab.json", "r", encoding="utf-8") as file:
    vocab = json.load(file)

    # vocab: dict[str|int, list[int]] from json
    bs_id, byte_list = max(vocab.items(), key=lambda kv: len(bytes(kv[1]).decode("utf-8", errors="replace")))
    biggest_string = bytes(byte_list).decode("utf-8", errors="replace")
    print(f"Biggest string: {biggest_string}")
    print(f"id: {bs_id}")
    print(f"len: {len(biggest_string)}")

    bb_id, biggest_bytes = max(vocab.items(), key= lambda kv: len(kv[1]))
    print(f"Biggest bytes: {biggest_bytes}")
    print(f"id: {bb_id}")
    print(f"len: {len(biggest_bytes)}\n\n")

    top = heapq.nlargest(
            10, vocab.items(), key=lambda kv: len(bytes(kv[1]).decode("utf-8", errors="replace"))
        )

    for i, (top_id, top_bt) in enumerate(top, start=1):
        s = bytes(top_bt).decode("utf-8", errors="replace")
        print(f"{i} biggest string: {s}")
        print(f"id: {top_id}")
        print(f"len: {len(s)}")
