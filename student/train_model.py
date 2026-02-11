from student.transformer import softmax
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entropy_loss(logits, targets):
    # implementing:
        # loss = -mean(x_target - log(sum of exp of logits))

    # logits is # (batch_size, vocab_size) targets is [batch_size]

    # subtract max value for numerical stability
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

    # compute log-softmax directly
    log_probs = shifted - log_sum_exp

    # gather correct class log-probabilities
    correct_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    return -correct_log_probs.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        other_params = {"lr": lr}
        super().__init__(params, other_params)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)


                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.

                m = state["m"]
                v = state["v"]
                m = beta1 * m + (1 - beta1 ) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                new_lr = lr * ((math.sqrt(1 - beta2**t)) / (1 - beta1**t))
                
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p

                p.data -= new_lr * (m / (torch.sqrt(v) + eps)) # Update weight tensor in-place.

                state["t"] = t + 1 # Increment iteration number.  
                state["m"] = m
                state["v"] = v
        return loss          



# TESTING SGD
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e3)
# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() 