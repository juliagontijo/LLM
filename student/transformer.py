import torch.nn as nn
import torch
import torch.nn.init as init
from einops import rearrange, einsum
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.dim_in = in_features
        self.dim_out = out_features
        self.device = device
        self.dtype = dtype

        # W shape is (d_out, d_in)
        self.W = nn.Parameter(torch.empty(self.dim_out, self.dim_in, device = device, dtype = dtype))
        std = math.sqrt(2.0 / (self.dim_in + self.dim_out))
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        # W shape is (d_out, d_in)
        # x shape is (... d_in)
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.vocab_size = num_embeddings
        self.embedding_dim = embedding_dim # = d_model
        self.device = device
        self.dtype = dtype

        self.hidden = nn.Parameter(torch.empty(self.vocab_size, self.embedding_dim, device=self.device, dtype = self.dtype))
        std = math.sqrt(1)
        init.trunc_normal_(self.hidden, mean=0.0, std=std, a=-3, b=3)


    def forward(self, token_ids):
        return self.hidden[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device=None, dtype = None):
        super().__init__()
        self.hidden_dim = d_model
        self.eps = eps 
        self.device = device
        self.dtype = dtype

        self.g = nn.Parameter(torch.empty(self.hidden_dim, device=self.device, dtype = self.dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        aux = x.pow(2) + self.eps
        rms = torch.sqrt(aux.mean(dim=-1, keepdim=True))
        result = (x/rms) * self.g

        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model))


    def forward(self, x):
        # x shape is ... d_model
        part1 = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        part3 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")  
        silu_ = self.silu(part1) * part3 # [... d_ff] * [... d_ff] = [... d_ff]
        out = einsum(self.w2, silu_, "d_model d_ff, ... d_ff -> ... d_model")

        return out 
    
    def silu(self, x):
        return x * torch.sigmoid(x) # no dim change

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k_lower_bound = self.d_k
        k_upper_bound = (self.d_k // 2)

        self.angle = torch.zeros(k_lower_bound, k_upper_bound, device = device)

        for i in range(k_lower_bound):
            for k in range(k_upper_bound):
                power = (2*k) / self.d_k
                self.angle[i][k] = i / (theta ** power)

        self.register_buffer("cos", torch.cos(self.angle), persistent=False)
        self.register_buffer("sin", torch.sin(self.angle), persistent=False)

        # print(f"\nCOSS:\n{self.cos}\n\n")
        # print(f"SIN:\n{self.sin}")

    def forward(self, x, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)

        # reshape tensor x from [... seq_len d_k] to [... seq_len (d_k//2) 2] to make pairs
        # from ([B, 12, 64]) to ([B, 12, 32, 2])
        x_pairs = rearrange(x, "... seq_len (d_k_2 d_2) -> ... seq_len d_k_2 d_2", d_k_2=self.d_k//2, d_2=2)

        # we have sin and cos for every 64 (max_seq_len) possible tokens. Our x has 12 tokens and we want to select tokens of indices stored in token_positions
        # so we chose from self.cos of shape (64, 32) only the 12 tokens of indice equals to i in token_positions, giving us a tensor of shape ([12,32])

        sliced_cos = self.cos[token_positions]
        sliced_sin = self.sin[token_positions]

        # sliced_cos = rearrange(sliced_cos, "seq dim -> 1 seq dim 1")
        # sliced_sin = rearrange(sliced_sin, "seq dim -> 1 seq dim 1")

        sliced_cos = self.cos[token_positions]   # [1, seq, dim]
        sliced_sin = self.sin[token_positions]   # [1, seq, dim]
        sliced_cos = sliced_cos.unsqueeze(-1)    # [1, seq, dim, 1]
        sliced_sin = sliced_sin.unsqueeze(-1)


        x0 = x_pairs[..., 0:1]
        x1 = x_pairs[..., 1:2]

        y0 = x0 * sliced_cos - x1 * sliced_sin
        y1 = x0 * sliced_sin + x1 * sliced_cos

        result = torch.cat([y0, y1], dim=-1)

        # reshape back from [... seq_len (d_k//2) 2] pairs to [... seq_len d_k]
        x = rearrange(result, "... seq_len d_k_2 d_2 -> ... seq_len (d_k_2 d_2)")
        return x
    
def softmax(x, i):
    x = x - x.max(dim=i, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(query, key, value, mask = None):
    # q, k has shape (batch_size, ..., seq_len, d_k)
    # v has shape (batch_size, ..., seq_lev, d_v)
    qk = einsum(query, key, " ... q d, ... k d -> ... q k")
    qk_scaled = qk / math.sqrt(key.shape[-1])
    
    if mask is not None:
        neg_inf = torch.finfo(qk_scaled.dtype).min
        qk_scaled = qk_scaled.where(mask, neg_inf)

    normalized = softmax(qk_scaled, -1)
    out = einsum(normalized, value, "... q k, ... k d -> ... q d")
    #output has shape (batch_size, ..., d_v)
    return out

class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, token_positions = None, rope = None, max_seq_length = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.head_dim = d_model // num_heads

        self.rope = rope
        self.token_positions = token_positions

        self.q_proj_weight = nn.Parameter(torch.zeros((self.num_heads* self.head_dim), self.d_model))
        self.k_proj_weight = nn.Parameter(torch.zeros((self.num_heads* self.head_dim), self.d_model))
        self.v_proj_weight = nn.Parameter(torch.zeros((self.num_heads* self.head_dim), self.d_model))
        self.o_proj_weight = nn.Parameter(torch.zeros(self.d_model, (self.num_heads * self.head_dim)))

    def forward(self, x):
        # x: [B, S, d_model]
        Q = einsum(x, self.q_proj_weight, "b s d_model, d_k d_model -> b s d_k")
        K = einsum(x, self.k_proj_weight, "b s d_model, d_k d_model -> b s d_k")
        V = einsum(x, self.v_proj_weight, "b s d_model, d_v d_model -> b s d_v")

        # split heads
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE per head (over head_dim), not over concatenated heads.
        if self.rope is not None:
            # rope_head = RotaryPositionalEmbedding(self.rope.theta, self.head_dim, self.rope.max_seq_len, device=x.device)
            Q = self.rope.forward(Q, self.token_positions)
            K = self.rope.forward(K, self.token_positions)

        # Decoder-only causal mask.
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, s, s]

        # attention per head (supports batch + head dims)
        O = scaled_dot_product_attention(Q, K, V, mask=mask)  # [b, h, s, d]

        # concat heads
        O = rearrange(O, "b h s d -> b s (h d)")

        # output projection
        out = einsum(O, self.o_proj_weight, "b s d_v, d_model d_v -> b s d_model")

        return out



# class FFN(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super().__init__()

#         self.fc1 = Linear(d_model, d_ff)
#         self.act = SwiGLU(d_model, d_ff)
#         self.fc2 = Linear(d_ff, d_model)

#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, token_positions = None, max_seq_length=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.max_seq_length = max_seq_length
        self.token_positions = token_positions

        self.ln1 = RMSNorm(d_model) # normalize
        self.attn = MHSA(d_model, num_heads, token_positions, rope, max_seq_length) 
        self.ffn = SwiGLU(d_model, d_ff) # FFN
        self.ln2 = RMSNorm(d_model) # normalize


    def forward(self, x):
        residual = x

        # normalize and mhsa with rope
        part1 = self.attn.forward(self.ln1.forward(x))

        # add result to residual
        intermediate_result = residual + part1

        # residual becomes this answer\
        residual = intermediate_result

        #normalize and FFN
        part2 = self.ffn(self.ln2.forward(intermediate_result))

        #add result to residual
        result = residual + part2

        return result


