"""
Streaming Label-Permuted EMNIST

Protocol based on Elsayed et al., 2024 (weight clipping paper):
- EMNIST balanced dataset, streamed one sample at a time
- Labels are randomly permuted every 2500 steps
- Network: 300x150 with LeakyReLU
- Metric: average online accuracy
- 5M total samples, 2500 tasks

Usage: python train_streaming_emnist.py
"""
import wandb
import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import datasets, transforms
from optimizers import HybridAdamFADE, HybridSGDFADE

PROJECT_NAME = "fade-streaming-emnist"

config_dict = dict(
    seed=64,
    algorithm="sgd_fade_head",  # "adam", "adamw", "sgd",
                       # "adam_fade_head", "sgd_fade_head",
                       # "adam_wc", "sgd_wc"
    for_lr='sgd',  # "adam" or "sgd", only relevant for FADE optimizers

    period=2500,  # steps between permutations
    num_samples=5000000,

    num_stable_classes=0,  # 0 means all classes permuted (default behavior, 
                           #                             we also use 24 for an additional
                           #                             experiment with partial permutation)

    # network
    hidden_sizes=[300, 150],  # matching Elsayed et al.
    activation='leaky_relu',  # 'relu' or 'leaky_relu'

    # training
    lr=0.005,

    # AdamW
    weight_decay=0.0,

    # Weight clipping
    kappa=2.0,

    # FADE params
    theta_wd=0.1,
    initial_gamma=-6.9,
    head_lr=None,  # defaults to lr if None

    # logging
    log_interval=2500,  # log once per task

    # device
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# ── Network ───────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation='leaky_relu'):
        super().__init__()
        act_fn = nn.LeakyReLU if activation == 'leaky_relu' else nn.ReLU

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_fn())
            in_size = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(self.features(x))


# ── Hooks ─────────────────────────────────────────────────────────────

def register_fade_hooks_classification(model):
    """Cache squared inputs and softmax correction for classification."""
    def hook_fn(module, input, output):
        with torch.no_grad():
            h = input[0].detach()
            
            # h^2 term
            if h.dim() == 1:
                h_sq = (h ** 2).view(1, -1)  # (1, hidden)
            else:
                h_sq = (h ** 2).mean(dim=0).view(1, -1)  # (1, hidden)
            
            # Softmax correction: p_k(1-p_k)
            logits = output.detach()
            p = torch.softmax(logits, dim=-1)
            pk_corr = p * (1.0 - p)  # (batch, num_classes)
            if pk_corr.dim() > 1:
                pk_corr = pk_corr.mean(dim=0)  # (num_classes,)
            
            # local_grad_sq has shape (num_classes, hidden) matching W
            module.weight.local_grad_sq = pk_corr.view(-1, 1) * h_sq
            
            if module.bias is not None:
                module.bias.local_grad_sq = pk_corr

    # Only apply to the head (last linear layer)
    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    head = linear_modules[-1]
    head.register_forward_hook(hook_fn)


# ── Weight Clipping ───────────────────────────────────────────────────

def weight_clip(model, kappa):
    """Clip weights to [-kappa * init_bound, kappa * init_bound] per layer."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = kappa / math.sqrt(fan_in)
                module.weight.data.clamp_(-bound, bound)
                if module.bias is not None:
                    module.bias.data.clamp_(-bound, bound)


# ── Build algorithm ───────────────────────────────────────────────────

def build_algorithm(config, model):
    name = config.algorithm
    head_lr = config.head_lr if config.head_lr is not None else config.lr
    hidden_params = list(model.features.parameters())
    head_params = list(model.head.parameters())

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)

    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)

    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)

    elif name == "adam_wc":
        return torch.optim.Adam(model.parameters(), lr=config.lr)

    elif name == "sgd_wc":
        return torch.optim.SGD(model.parameters(), lr=config.lr)

    elif name == "sgd_fade_head":
        return HybridSGDFADE(hidden_params, head_params, lr=config.lr,
                          head_lr=head_lr, theta_wd=config.theta_wd,
                          initial_gamma=config.initial_gamma,
                          for_lr=config.for_lr)

    elif name == "adam_fade_head":
        return HybridAdamFADE(hidden_params, head_params, lr=config.lr,
                               head_lr=head_lr, theta_wd=config.theta_wd,
                               initial_gamma=config.initial_gamma,
                               for_lr=config.for_lr)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


# ── Data ──────────────────────────────────────────────────────────────

def load_dataset(dataset_name):
    """Load dataset and return (images, labels, num_classes, input_size)."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    if dataset_name == 'emnist':
        dataset = datasets.EMNIST('./data', split='balanced', train=True,
                                   download=True, transform=transform)
        num_classes = 47
        input_size = 784
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return images, labels, num_classes, input_size


def make_label_permutation(num_classes, rng):
    """Create a random label permutation."""
    perm = rng.permutation(num_classes)
    return torch.from_numpy(perm).long()


def make_partial_label_permutation(num_classes, stable_classes, rng):
    """Permute labels only for non-stable classes."""
    perm = torch.arange(num_classes)
    permuted_classes = [c for c in range(num_classes) if c not in stable_classes]
    sub_perm = rng.permutation(len(permuted_classes))
    for i, new_i in enumerate(sub_perm):
        perm[permuted_classes[i]] = permuted_classes[new_i]
    return perm.long()


# ── Diagnostics ───────────────────────────────────────────────────────

def compute_weight_norm(model):
    """Compute l2 norm of all weights."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += (p ** 2).sum().item()
    return math.sqrt(total)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    wandb.init(project=PROJECT_NAME, config=config_dict)
    config = wandb.config
    device = config.device
    name = config.algorithm

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.RandomState(config.seed)

    # Load dataset
    images, true_labels, num_classes, input_size = load_dataset('emnist')
    images = images.to(device)
    true_labels = true_labels.to(device)
    n_data = len(images)

    # Build model
    model = MLP(input_size, config.hidden_sizes, num_classes,
                activation=config.activation).to(device)
    
    if name in ("adam_fade_head", "sgd_fade_head"):
        register_fade_hooks_classification(model)

    opt = build_algorithm(config, model)
    loss_fn = nn.CrossEntropyLoss()

    use_wc = name.endswith('_wc')

    # Stable classes for partial permutation
    if config.num_stable_classes > 0:
        stable_rng = np.random.RandomState(config.seed)
        stable_classes = set(stable_rng.choice(num_classes, config.num_stable_classes, replace=False))

    # Initialize permutation
    if config.num_stable_classes > 0:
        label_perm = make_partial_label_permutation(num_classes, stable_classes, rng).to(device)
    else:
        label_perm = make_label_permutation(num_classes, rng).to(device)

    # Tracking
    correct_buffer = []
    task_accuracies = []
    total_correct = 0
    total_steps = 0

    for step in range(config.num_samples):
        # New permutation every period steps
        if step > 0 and step % config.period == 0:
            if config.num_stable_classes > 0:
                label_perm = make_partial_label_permutation(num_classes, stable_classes, rng).to(device)
            else:
                label_perm = make_label_permutation(num_classes, rng).to(device)

        # Sample one data point uniformly at random
        idx = rng.randint(0, n_data)
        x = images[idx:idx+1]  # (1, C, H, W)
        y_true = true_labels[idx:idx+1]

        # Apply permutation
        y = label_perm[y_true].to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)
        correct = (pred == y).float().item()

        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        correct_buffer.append(correct)
        total_correct += correct
        total_steps += 1

        # Weight clipping after optimizer step
        if use_wc:
            weight_clip(model, config.kappa)

        # Logging
        if (step + 1) % config.log_interval == 0:
            task_acc = np.mean(correct_buffer)
            task_accuracies.append(task_acc)

            log_dict = {
                "step": step + 1,
                "task": (step + 1) // config.period,
                "online_accuracy": task_acc,
                "cumulative_accuracy": total_correct / total_steps if total_steps > 0 else 0,
                "weight_norm": compute_weight_norm(model),
            }

            # Head weight magnitude
            with torch.no_grad():
                log_dict["head_weight_magnitude"] = model.head.weight.abs().mean().item()

            # FADE lambdas
            if name in ("adam_fade_head", "sgd_fade_head"):
                head_lambdas = torch.exp(opt.head_opt.gamma[0]).detach().cpu()
                log_dict["lambda_head_mean"] = head_lambdas.mean().item()
                log_dict["lambda_head_max"] = head_lambdas.max().item()
                log_dict["lambda_head_min"] = head_lambdas.min().item()

            wandb.log(log_dict)
            correct_buffer = []

            task_num = (step + 1) // config.period
            print(f"  Task {task_num:4d} | online_acc={task_acc:.3f} | "
                  f"cumul_acc={total_correct / total_steps:.3f} | "
                  f"w_norm={log_dict['weight_norm']:.1f}")

    # Summary
    avg_acc = total_correct / total_steps if total_steps > 0 else 0
    last_quarter_acc = np.mean(task_accuracies[3 * len(task_accuracies) // 4:]) if len(task_accuracies) > 0 else 0
    last10_task_acc = np.mean(task_accuracies[-10:]) if len(task_accuracies) >= 10 else np.mean(task_accuracies)

    wandb.summary["avg_online_accuracy"] = avg_acc
    wandb.summary["last_quarter_accuracy"] = last_quarter_acc
    wandb.summary["last10_task_accuracy"] = last10_task_acc

    wandb.finish()
    print(f"\nDone. {name}: avg_acc={avg_acc:.3f} "
          f"last_quarter={last_quarter_acc:.3f} "
          f"last10_tasks={last10_task_acc:.3f}")


if __name__ == "__main__":
    main()