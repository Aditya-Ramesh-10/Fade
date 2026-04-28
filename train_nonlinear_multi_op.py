"""
Nonlinear Continual Learning

Protocol:
- Teacher NN with K output targets
- Student NN learns online (single sample per step)
- Teacher flips output neuron weights periodically

Usage: python3 train_nonlinear_multi_op.py
"""

import wandb
import torch
import torch.nn as nn
import numpy as np
import math
from optimizers import HybridAdamFADE, HybridSGDFADE

PROJECT_NAME = "fade-nonlinear-tracking"

config_dict = dict(
    seed=42,
    algorithm="sgd_fade_head",  # "adam", "adamw", "sgd",
                                 # "sgd_fade_head", "adam_fade_head"
    for_lr='sgd',  # "adam" or "sgd", only relevant for FADE optimizers, 
                   # we used "sgd" for sgd_fade_head and "adam" for adam_fade_head

    # problem specification
    period=500,
    flip_slow_period_mult=15,
    m_total=32,          # input dimension
    teacher_hidden=256,
    num_outputs=20,      # K: number of output targets

    # output group fractions (must sum to 1.0)
    frac_stable=0.30,    # never change
    frac_fast=0.35,      # flip every period
    frac_slow=0.35,      # flip every slow_period_mult * period

    # student network
    student_hidden=256,

    # training
    num_samples=2000000,
    lr=0.01,

    # AdamW
    weight_decay=0.0,

    # FADE params
    theta_wd=2.0,
    initial_gamma=-9.2, # corresponds to initial lambda of 0.001
    head_lr=None,  # defaults to lr if None

    # logging
    log_interval=500,
    error_over_last=500000,

    # device
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# ── Teacher ───────────────────────────────────────────────────────────

class TeacherNet(nn.Module):
    """Teacher with K outputs grouped into stable/fast/slow non-stationarity.
    When a group flips, ALL weights for each output neuron in that group
    are independently multiplied by a random sign (±1 with equal probability).
    """
    def __init__(self, m_total, hidden_size, num_outputs,
                 frac_stable=0.25, frac_fast=0.50, frac_slow=0.25,
                 num_hidden_layers=1):
        super().__init__()
        self.input_layer = nn.Linear(m_total, hidden_size, bias=False)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_hidden_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_size, num_outputs, bias=False)

        # Assign output neurons to groups
        perm = torch.randperm(num_outputs)
        n_stable = max(1, int(num_outputs * frac_stable))
        n_fast = max(1, int(num_outputs * frac_fast))
        n_slow = num_outputs - n_stable - n_fast

        self.stable_outputs = perm[:n_stable]
        self.fast_outputs = perm[n_stable:n_stable + n_fast]
        self.slow_outputs = perm[n_stable + n_fast:]

        self.n_stable = n_stable
        self.n_fast = n_fast
        self.n_slow = n_slow

        fan_in = self.output_layer.in_features
        self.output_bound = 1.0 / math.sqrt(fan_in)

    def flip_fast_outputs(self):
        with torch.no_grad():
            mask = torch.randint(0, 2, size=self.output_layer.weight[self.fast_outputs].shape, device=self.output_layer.weight.device) * 2 - 1
            self.output_layer.weight[self.fast_outputs] *= mask

    def flip_slow_outputs(self):
        with torch.no_grad():
            mask = torch.randint(0, 2, size=self.output_layer.weight[self.slow_outputs].shape, device=self.output_layer.weight.device) * 2 - 1
            self.output_layer.weight[self.slow_outputs] *= mask

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.output_layer(h)


# ── Student ───────────────────────────────────────────────────────────

class StudentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, num_hidden_layers=1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        h = self.features(x)
        return self.head(h)


# ── Hooks ─────────────────────────────────────────────────────────────

def register_fade_hooks(model):
    """Cache squared inputs for every Linear layer."""
    def hook_fn(module, input, output):
        with torch.no_grad():
            h = input[0].detach()
            if h.dim() == 1:
                module.weight.local_grad_sq = (h ** 2).view(1, -1)
            else:
                module.weight.local_grad_sq = (h ** 2).mean(dim=0).view(1, -1)
            if module.bias is not None:
                module.bias.local_grad_sq = torch.ones_like(module.bias)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(hook_fn)



# ── Diagnostics ───────────────────────────────────────────────────────

def compute_weight_norm(model):
    """Compute l2 norm of all weights."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += (p ** 2).sum().item()
    return math.sqrt(total)


# ── Build algorithm ───────────────────────────────────────────────────

def build_algorithm(config, model):
    name = config.algorithm
    head_lr = config.head_lr if config.head_lr is not None else config.lr
    hidden_params = list(model.features.parameters())
    head_params = list(model.head.parameters())

    effective_head_lr = head_lr
    sgd_lr = config.lr

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)

    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=sgd_lr,
                               weight_decay=config.weight_decay)

    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    elif name == "sgd_fade_head":
        return HybridSGDFADE(hidden_params, head_params, lr=sgd_lr,
                             head_lr=effective_head_lr, theta_wd=config.theta_wd,
                             initial_gamma=config.initial_gamma,
                             for_lr=config.for_lr)

    elif name == "adam_fade_head":
        return HybridAdamFADE(hidden_params, head_params, lr=config.lr,
                              head_lr=effective_head_lr, theta_wd=config.theta_wd,
                              initial_gamma=config.initial_gamma,
                              for_lr=config.for_lr)

    else:
        raise ValueError(f"Unknown algorithm: {name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    wandb.init(project=PROJECT_NAME, config=config_dict)
    config = wandb.config
    device = config.device
    name = config.algorithm

    input_size = config.m_total
    K = config.num_outputs

    # Teacher
    torch.manual_seed(config.seed)
    teacher = TeacherNet(input_size, config.teacher_hidden, K,
                         frac_stable=config.frac_stable,
                         frac_fast=config.frac_fast,
                         frac_slow=config.frac_slow,
                         num_hidden_layers=1).to(device)

    print(f"Teacher output groups: {teacher.n_stable} stable, "
          f"{teacher.n_fast} fast (every {config.period}), "
          f"{teacher.n_slow} slow (every {config.flip_slow_period_mult * config.period})")

    # Student
    torch.manual_seed(config.seed + 1000)
    student = StudentNet(input_size,
                         config.student_hidden, K,
                         num_hidden_layers=1).to(device)

    register_fade_hooks(student)

    # loss function for exact meta-gradient 1/2 * SSE
    loss_fn = lambda y, t: 0.5 * nn.functional.mse_loss(y, t, reduction='sum')

    opt = build_algorithm(config, student)

    total_error = 0.0
    total_steps = 0
    tail_error = 0.0
    tail_steps = 0
    tail_start = config.num_samples - config.error_over_last

    error_acc = torch.zeros(1, device=device)
    error_acc_stable = torch.zeros(1, device=device)
    error_acc_fast = torch.zeros(1, device=device)
    error_acc_slow = torch.zeros(1, device=device)
    steps_since_log = 0
    
    chunk_size = 10000
    x_buffer = torch.randn(chunk_size, config.m_total, device=device)
    buf_idx = 0


    for step in range(config.num_samples):

        if buf_idx >= chunk_size:
            x_buffer = torch.randn(chunk_size, config.m_total, device=device)
            buf_idx = 0
        x = x_buffer[buf_idx]
        buf_idx += 1

        y_target = teacher(x)
        y_pred = student(x)
        loss = loss_fn(y_pred, y_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            error_acc += loss.detach()
            diff = (y_pred - y_target) ** 2
            error_acc_stable += diff[teacher.stable_outputs].mean()
            error_acc_fast += diff[teacher.fast_outputs].mean()
            error_acc_slow += diff[teacher.slow_outputs].mean()
        steps_since_log += 1

        # Logging
        if steps_since_log % config.log_interval == 0:

            mse_val = error_acc.item() / steps_since_log
            mse_val = 2 * mse_val / config.num_outputs  # convert back from 1/2 SSE to MSE for logging
            total_error += error_acc.item()
            total_steps += steps_since_log
            if step >= tail_start:
                tail_error += error_acc.item()
                tail_steps += steps_since_log
            log_dict = {
                "step": step + 1,
                "mse": mse_val,
                "mse_stable": error_acc_stable.item() / steps_since_log if steps_since_log > 0 else 0,
                "mse_fast": error_acc_fast.item() / steps_since_log if steps_since_log > 0 else 0,
                "mse_slow": error_acc_slow.item() / steps_since_log if steps_since_log > 0 else 0,
                "weight_norm": compute_weight_norm(student),
            }

            with torch.no_grad():
                log_dict["head_weight_magnitude"] = student.head.weight.abs().mean().item()

            # FADE lambdas
            if name in ("adam_fade_head", "adamw_fade_head", "adam_fade_idbd_head", "sgd_fade_head"):
                head_lambdas = torch.exp(opt.head_opt.gamma[0]).detach().cpu()
                log_dict["lambda_head_mean"] = head_lambdas.mean().item()
                log_dict["lambda_head_max"] = head_lambdas.max().item()
                log_dict["lambda_head_min"] = head_lambdas.min().item()

                # Per-group lambda means (rows of head weight matrix)
                if head_lambdas.dim() == 2:  # shape: (K, hidden)
                    log_dict["lambda_stable_mean"] = head_lambdas[teacher.stable_outputs].mean().item()
                    log_dict["lambda_fast_mean"] = head_lambdas[teacher.fast_outputs].mean().item()
                    if teacher.n_slow > 0:
                        log_dict["lambda_slow_mean"] = head_lambdas[teacher.slow_outputs].mean().item()

            wandb.log(log_dict)
            error_acc = torch.zeros(1, device=device)
            error_acc_stable = torch.zeros(1, device=device)
            error_acc_fast = torch.zeros(1, device=device)
            error_acc_slow = torch.zeros(1, device=device)
            steps_since_log = 0

            print(f"  Step {step + 1:7d} | mse={log_dict['mse']:.6f} "
                  f"(stable={log_dict['mse_stable']:.6f} "
                  f"fast={log_dict['mse_fast']:.6f} "
                  f"slow={log_dict['mse_slow']:.6f}) | "
                  f"w_norm={log_dict['weight_norm']:.1f}")

        # Teacher non-stationarity
        if (step + 1) % config.period == 0:
            teacher.flip_fast_outputs()
            if (step + 1) % (config.flip_slow_period_mult * config.period) == 0:
                teacher.flip_slow_outputs()

    # Summary
    asymptotic_mse = tail_error / tail_steps if tail_steps > 0 else 0
    asymptotic_mse = 2 * asymptotic_mse / config.num_outputs  # convert back from 1/2 SSE to MSE
    full_mse = total_error / total_steps
    full_mse = 2 * full_mse / config.num_outputs  # convert back from 1/2 SSE to MSE

    wandb.summary["asymptotic_mse"] = asymptotic_mse
    wandb.summary["full_mse"] = full_mse

    wandb.finish()
    print(f"\nDone. {name}: asymptotic_mse={asymptotic_mse:.4f} full_mse={full_mse:.4f}")


if __name__ == "__main__":
    main()