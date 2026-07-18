# -*- coding: utf-8 -*-
"""Muon optimizer (Moonshot / Keller Jordan), single-device, self-contained.

Muon = momentum + Newton-Schulz orthogonalization of the update, applied to 2D
hidden weights; everything else (1D: norms, biases) falls back to AdamW. We use
Moonshot's "Muon is scalable" update-RMS scaling (0.2*sqrt(max(fan))) so the SAME
base LR that AdamW uses is a fair drop-in (no separate Muon LR sweep needed for a
first-pass comparison). Faithful vendored copy — no external dep on the pod.
"""
import math
import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Quintic Newton-Schulz iteration -> approx orthogonalization of G (2D)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)          # standard coefficients
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):                     # NS assumes rows <= cols
        X = X.T; transposed = True
    X = X / (X.norm() + 1e-7)                      # spectral-norm <= 1 guarantee
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """muon_params: 2D weights (get Muon). adamw_params: the rest (get AdamW)."""
    def __init__(self, muon_params, lr=1e-4, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_params=None, adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                        weight_decay=weight_decay)
        muon_params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        super().__init__(muon_params + adamw_params, defaults)
        for p in muon_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            mom, nesterov, ns = group["momentum"], group["nesterov"], group["ns_steps"]
            b1, b2 = group["adamw_betas"]; eps = group["adamw_eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if st.get("use_muon", False):
                    g = p.grad
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf = st["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    g = g.add(buf, alpha=mom) if nesterov else buf
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    u = zeropower_via_newtonschulz5(g, ns).to(p.dtype)
                    # Moonshot update-RMS scaling: makes base lr comparable to AdamW
                    adj_lr = lr * 0.2 * math.sqrt(max(g.size(0), g.size(1)))
                    if wd:                            # NOTE: decay uses base lr, not adj_lr;
                        p.data.mul_(1 - lr * wd)      # with wd>0 retune it (adj_lr is ~64-137x lr)
                    p.data.add_(u.view_as(p), alpha=-adj_lr)
                else:                                   # AdamW fallback (1D params)
                    g = p.grad
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    if wd:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(m, denom, value=-lr / bc1)
        return loss
