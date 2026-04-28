import torch
import torch.nn as nn
import math


class FADE_Optimizer:
    """Per-parameter adaptive decoupled weight decay"""
    def __init__(self, params, lr=1e-3, theta_lambda=1e-2,
                 initial_gamma=-2.3, gamma_min=-15.0, gamma_max=0.0,
                 for_lr='sgd'):
        self.params = list(params)
        self.lr = lr

        self.for_lr = for_lr
        
        if for_lr == 'adam':
            self.base_optimizer = torch.optim.Adam(self.params, lr=lr)
        elif for_lr == 'sgd':
            self.base_optimizer = torch.optim.SGD(self.params, lr=lr)
        else:
            raise ValueError(f"Unsupported for_lr value: {for_lr}")
        
        self.theta_lambda = theta_lambda
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        self.gamma = [torch.full_like(p, initial_gamma) for p in self.params]
        self.g_trace = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            neg_g = -grad
            w_old = p.data.clone()

            self.gamma[i].add_(self.theta_lambda * neg_g * self.g_trace[i]).clamp_(
                self.gamma_min, self.gamma_max)
            
            lam = torch.exp(self.gamma[i])

            if hasattr(p, 'local_grad_sq'):
                local_grad_sq = p.local_grad_sq
            else:
                local_grad_sq = 0.0 

            if self.for_lr == 'adam':
                state = self.base_optimizer.state.get(p, None)
                if state and 'exp_avg_sq' in state:
                    beta2 = self.base_optimizer.defaults['betas'][1]
                    eps = self.base_optimizer.defaults['eps']
                    step_count = state['step']
                    bc = 1 - beta2 ** step_count
                    v_hat = state['exp_avg_sq'] / bc
                    effective_alpha = self.lr / (torch.sqrt(v_hat) + eps)
                    decay_factor = (1 - lam - effective_alpha * local_grad_sq).clamp(min=0)
                else:
                    # First step, no state yet — fall back to lr
                    decay_factor = (1.0 - lam - self.lr * local_grad_sq).clamp(min=0.0)
            else:    
                decay_factor = (1.0 - lam - self.lr * local_grad_sq).clamp(min=0.0)

            self.g_trace[i].mul_(decay_factor).sub_(lam * w_old)

            p.data.mul_(1.0 - lam)

        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class HybridAdamFADE:
    """Adam on hidden layers, FADE on head."""
    def __init__(self, hidden_params, head_params, lr=1e-3,
                 head_lr=1e-3, theta_wd=1e-2, initial_gamma=-2.3,
                 for_lr='adam'):
        self.hidden_opt = torch.optim.Adam(hidden_params, lr=lr)
        self.head_opt = FADE_Optimizer(head_params, lr=head_lr,
                                        theta_lambda=theta_wd,
                                        initial_gamma=initial_gamma,
                                        for_lr=for_lr)
    def step(self):
        self.hidden_opt.step()
        self.head_opt.step()

    def zero_grad(self):
        self.hidden_opt.zero_grad()
        self.head_opt.zero_grad()


class HybridSGDFADE:
    """SGD on hidden layers, FADE on head."""
    def __init__(self, hidden_params, head_params, lr=1e-3,
                 head_lr=1e-3, theta_wd=1e-2, initial_gamma=-2.3,
                 for_lr='sgd'):
        self.hidden_opt = torch.optim.SGD(hidden_params, lr=lr)
        self.head_opt = FADE_Optimizer(head_params, lr=head_lr,
                                        theta_lambda=theta_wd,
                                        initial_gamma=initial_gamma,
                                        for_lr=for_lr)
    def step(self):
        self.hidden_opt.step()
        self.head_opt.step()

    def zero_grad(self):
        self.hidden_opt.zero_grad()
        self.head_opt.zero_grad()