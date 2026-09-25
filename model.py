"""
Mamba from Scratch: Selective State Spaces

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rms_norm
def rms_norm(x, weight, eps=1e-5):
    """Normalize a hidden sequence with RMSNorm using a learned per-channel scale."""
    # TODO: Normalize a hidden sequence with RMSNorm using a learned per-channel scale...
    return x * weight / torch.sqrt((x**2).mean(dim=-1, keepdim=True)+eps)

# Step 2 - silu
def silu(x):
    """Apply the SiLU activation elementwise."""
    # TODO: Implement `silu` so that it applies the SiLU activation to a float tensor of any shape.
    return torch.nn.functional.silu(x)

# Step 3 - causal_depthwise_conv1d
def causal_depthwise_conv1d(x, weight, bias=None):
    """Run a causal depthwise 1-D convolution over a (B, L, E) sequence.

    Args:
        x: (B, L, E) input sequence.
        weight: (E, K) per-channel kernel.
        bias: optional (E,) added after the convolution.

    Returns:
        (B, L, E) output sequence.
    """

    # TODO: Implement causal_depthwise_conv1d to produce a causally convolved sequence of the same length.
    E, K = weight.shape
    xnew = x.transpose(1, 2)
    wnew = weight.unsqueeze(1)
    pad = torch.nn.functional.pad(xnew, (K-1, 0))
    trans = torch.nn.functional.conv1d(pad, wnew, bias=bias, groups=E)
    return trans.transpose(1, 2)

# Step 4 - in_proj_split
def in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    # TODO: Project a token sequence to expanded width and split into SSM input x and gate z.
    proj = torch.nn.functional.linear(u, weight, bias = bias)
    E = proj.shape[-1] // 2 
    x, z = torch.split(proj, E, dim=-1)
    return x, z

# Step 5 - compute_delta
def compute_delta(x, weight, bias=None):
    """Compute a strictly positive per-token timestep Delta.

    x: (B, L, E), weight: (E, E) nn.Linear layout, bias: optional (E,).
    Returns delta of shape (B, L, E).
    """
    # TODO: Implement compute_delta to produce a strictly positive per-token timestep Delta.
    out = torch.nn.functional.linear(x, weight, bias=bias)
    return torch.nn.functional.softplus(out)

# Step 6 - project_bc
def project_bc(x, weight_b, weight_c):
    """Project the SSM input to input-dependent B and C state vectors of size N."""
    # TODO: Map an SSM input sequence to a pair of input-dependent B and C state vectors...
    return x @ weight_b.T, x @ weight_c.T

# Step 7 - make_diagonal_a
def make_diagonal_a(log_a):
    """Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A."""
    # TODO: Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A....
    return -torch.exp(log_a)

# Step 8 - discretize_a_zoh
def discretize_a_zoh(delta, a):
    """Discretize a diagonal continuous state matrix with zero-order hold.

    delta: torch tensor of shape (..., d)
    a: torch tensor of shape (d, n)
    Returns a_bar of shape (..., d, n).
    """
    # TODO: Implement `discretize_a_zoh` to discretize a diagonal state matrix with zero-order hold.
    dnew = delta.unsqueeze(-1)
    return torch.exp(dnew*a)

# Step 9 - discretize_b_zoh
def discretize_b_zoh(delta, a, b):
    """Discretize B with the exact diagonal zero-order-hold formula.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        b_bar: (batch, seq_len, d_inner, d_state) discrete B.
    """
    # TODO: Convert continuous B into discrete B_bar with the exact diagonal ZOH formula...
    d = delta.unsqueeze(-1)
    bnew = b.unsqueeze(2)
    return (torch.exp(d*a)-1)/a * bnew

# Step 10 - compare_euler_zoh_b
def compare_euler_zoh_b(delta, a, b):
    """Compare exact ZOH discrete B to the Euler shortcut.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        dict with keys 'b_bar_zoh', 'b_bar_euler', and 'abs_diff', each
        of shape (batch, seq_len, d_inner, d_state).
    """
    # TODO: Compare exact ZOH discrete B against the Euler shortcut.
    zoh = discretize_b_zoh(delta, a, b)
    delta = delta.unsqueeze(-1)
    b = b.unsqueeze(2)
    euler = delta*b
    abs_diff = torch.abs(zoh-euler)
    return {"b_bar_zoh": zoh, 
    "b_bar_euler": euler,
    "abs_diff":abs_diff}

# Step 11 - siso_state_update
def siso_state_update(h_prev, a_bar, b_bar, c, x_t):
    """Apply one SISO state update and return the scalar readout."""
    # TODO: Apply one SISO state update and return the scalar readout...
    h_t = a_bar * h_prev + b_bar * x_t 
    y_t = (c * h_t).sum()
    return y_t, h_t

# Step 12 - scan_single_channel
import torch


def scan_single_channel(x, a_bar, b_bar, c, h0=None):
    """Scan a single channel sequentially over time and return both the outputs and the final hidden state."""
    L = x.shape[0]
    N = a_bar.shape[-1]

    if h0 is None:
        h0 = torch.zeros(N, dtype=x.dtype, device=x.device)

    ys = []
    h_prev = h0

    if L == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device), h0

    for t in range(L):
        # Step through time sequentially using the single-step recurrence
        y_t, h_t = siso_state_update(h_prev, a_bar[t], b_bar[t], c[t], x[t])
        ys.append(y_t)
        h_prev = h_t
   
    # Stack collected outputs across the time dimension: shape (L,)
    y = torch.stack(ys, dim=0)

    return y, h_prev

# Step 13 - selective_scan
import torch


def selective_scan(x, a_bar, b_bar, c, h0=None):
    """Run a selective scan over a batched multi-channel sequence."""
    B, L, E = x.shape
    N = a_bar.shape[-1]

    # Preallocate output sequence and final hidden state
    y = torch.zeros((B, L, E), dtype=x.dtype, device=x.device)
    h_final = torch.zeros((B, E, N), dtype=x.dtype, device=x.device)

    # Handle sequence length 0 edge-case
    if L == 0:
        if h0 is not None:
            h_final.copy_(h0)
        return y, h_final

    # Iterate over batch elements and channels
    for b in range(B):
        # In standard Mamba/selective SSMs, c is often shared across channels: (B, L, N)
        # If c is per-channel: (B, L, E, N), adjust slicing accordingly: c[b, :, e, :]
        c_b = c[b] if c.ndim == 3 else None

        for e in range(E):
            # Extract 1D sequence for current batch and channel: (L,)
            x_be = x[b, :, e]

            # Extract parameter slices: (L, N)
            a_bar_be = a_bar[b, :, e, :]
            b_bar_be = b_bar[b, :, e, :]
            c_be = c_b if c_b is not None else c[b, :, e, :]

            # Extract initial hidden state for this channel if provided: (N,)
            h0_be = h0[b, e, :] if h0 is not None else None

            # Process single channel
            y_be, h_be = scan_single_channel(
                x_be, a_bar_be, b_bar_be, c_be, h0=h0_be
            )

            # Store results
            y[b, :, e] = y_be
            h_final[b, e] = h_be

    return y, h_final

# Step 14 - compare_constant_vs_selective_delta
def compare_constant_vs_selective_delta(x, a, b, c, delta_const, delta_sel):
    """Compare SSM scan outputs under a constant Delta versus a selective Delta.

    x: (batch, seq_len, d_inner)
    a: (d_inner, d_state) strictly negative continuous diagonal A
    b: (batch, seq_len, d_state)
    c: (batch, seq_len, d_state)
    delta_const: (batch, seq_len, d_inner) non-selective timestep
    delta_sel: (batch, seq_len, d_inner) input-dependent timestep

    Returns:
        y_const: (batch, seq_len, d_inner)
        y_sel: (batch, seq_len, d_inner)
    """
    # TODO: Compare SSM scan outputs under a constant Delta versus a selective Delta...
    a_bar_c = discretize_a_zoh(delta_const, a)
    b_bar_c = discretize_b_zoh(delta_const, a, b)
    a_bar_s = discretize_a_zoh(delta_sel, a)
    b_bar_s = discretize_b_zoh(delta_sel, a, b)

    y_c, _ = selective_scan(x, a_bar_c, b_bar_c, c, h0=None)
    y_s, _ = selective_scan(x, a_bar_s, b_bar_s, c, h0=None)
    return y_c, y_s

# Step 15 - gate_scan_output
def gate_scan_output(y, z):
    """Modulate the selective-scan output y by the parallel gate branch z."""
    # TODO: Modulate the selective-scan output y by the parallel gate branch z.
    return silu(z)*y

# Step 16 - out_proj
def out_proj(y, weight, bias=None):
    """Project gated scan output from d_inner back to d_model.

    y: (..., d_inner)
    weight: (d_model, d_inner)
    bias: (d_model,) or None
    Returns: (..., d_model)
    """
    # TODO: Implement out_proj, the linear map that sends the gated SSM scan back to model width.
    return torch.nn.functional.linear(y, weight, bias)

# Step 17 - mamba_mixer
def mamba_mixer(u, params):
    """Run one full Mamba selective-SSM mixer on a token sequence.

    Args:
        u: (B, L, D) input sequence.
        params: dict of mixer weights.

    Returns:
        (B, L, D) mixer output.
    """
    # 1. Unpack required weights
    in_proj_weight = params["in_proj_weight"]
    conv_weight = params["conv_weight"]
    dt_weight = params["dt_weight"]
    weight_b = params["weight_b"]
    weight_c = params["weight_c"]
    log_a = params["log_a"]
    out_proj_weight = params["out_proj_weight"]

    # Optional biases
    in_proj_bias = params.get("in_proj_bias")
    conv_bias = params.get("conv_bias")
    dt_bias = params.get("dt_bias")
    out_proj_bias = params.get("out_proj_bias")

    # 2. Input projection: split u into SSM input branch (x) and gating branch (z)
    x, z = in_proj_split(u, in_proj_weight, bias=in_proj_bias)

    # 3. Causal depthwise 1D conv + SiLU activation on x
    x = causal_depthwise_conv1d(x, conv_weight, bias=conv_bias)
    x = silu(x)

    # 4. Compute input-dependent SSM parameters (Delta, B, C)
    delta = compute_delta(x, dt_weight, bias=dt_bias)
    b, c = project_bc(x, weight_b, weight_c)

    # 5. Build continuous A and discretize (Delta, A, B) via ZOH
    a = make_diagonal_a(log_a)
    a_bar = discretize_a_zoh(delta, a)
    b_bar = discretize_b_zoh(delta, a, b)

    # 6. Run selective scan (discard the final SSM state)
    y, _ = selective_scan(x, a_bar, b_bar, c, h0=None)

    # 7. Gate with the z branch and project back to dimension D
    y = gate_scan_output(y, z)
    out = out_proj(y, out_proj_weight, bias=out_proj_bias)

    return out

# Step 18 - mamba_block
def mamba_block(x, params):
    """Apply a pre-norm residual Mamba block to a token sequence.

    Args:
        x: (B, L, D) hidden sequence.
        params: dict with norm_weight (D,) plus every mamba_mixer key.

    Returns:
        (B, L, D) block output.
    """
    # TODO: Wrap the selective mixer in a pre-norm residual block...
    norm_weight = params["norm_weight"]
    x_new = rms_norm(x, norm_weight)
    out = mamba_mixer(x_new, params)
    return x + out

# Step 19 - run_mamba_lm_stack
def run_mamba_lm_stack(embeddings, params):
    """Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.

    Args:
        embeddings: (B, L, D) token embeddings.
        params: dict with key `blocks` (list of per-block dicts for `mamba_block`)
            and key `norm_weight` of shape (D,) for the final RMSNorm (eps=1e-5).

    Returns:
        (B, L, D) hidden states after the stack and final RMSNorm.
    """
    # TODO: Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.
    x = embeddings 
    for p in params["blocks"]:
        x = mamba_block(x, p)
    return rms_norm(x, params["norm_weight"])

# Step 20 - mamba_lm_forward
def mamba_lm_forward(token_ids, params):
    """Map token ids through embeddings, the Mamba stack, and an LM head.

    Args:
        token_ids: (B, L) integer tensor of token ids.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            blocks (list), and norm_weight (D,).

    Returns:
        (B, L, V) logits.
    """
    # TODO: Map a batch of token ids to next-token logits over the vocabulary...
    emb = torch.nn.functional.embedding(token_ids, params['embed_weight'])
    out = run_mamba_lm_stack(emb, params)
    return torch.nn.functional.linear(out, params["lm_head_weight"])

# Step 21 - next_token_cross_entropy
def next_token_cross_entropy(logits, token_ids):
    """Compute the mean next-token cross-entropy from logits and token ids."""
    # TODO: Compute the mean next-token cross-entropy loss...
    B, T, V = logits.shape
    log = logits[:, :-1, :].reshape(-1, V)
    tok = token_ids[:, 1:].reshape(-1).long()
    return torch.nn.functional.cross_entropy(log, tok)

# Step 22 - sgd_training_step (not yet solved)
# TODO: implement

# Step 23 - mamba_recurrent_step (not yet solved)
# TODO: implement

# Step 24 - greedy_generate (not yet solved)
# TODO: implement

# Step 25 - train_tiny_mamba_and_generate (not yet solved)
# TODO: implement

