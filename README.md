# Forgetting through Adaptive DEcay (FADE)

The code in this repository supplements the paper ["Learning to Forget: Continual Learning with Adaptive Weight Decay"](http://arxiv.org/abs/2604.27063).


## Repository Structure

* **`algs.py`**: Contains the linear implementations of FADE and baseline algorithms.
* **`optimizers.py`**: Contains the PyTorch optimizer implementations for neural networks (`FADE_Optimizer`, `HybridAdamFADE`, `HybridSGDFADE`).
* **`data_generators.py`**: Contains the synthetic data generator (`PeriodicLinearParameterFlip`) for the linear tracking task.
* **`train_linear.py`**: Training script for the linear tracking problem.
* **`train_nonlinear_multi_op.py`**: Training script for the nonlinear teacher-student tracking problem.
* **`train_permuted_emnist.py`**: Training script for the streaming label-permuted EMNIST classification problem.

## Instructions

You can reproduce the experiments from the paper by running the respective training scripts. Hyperparameters are configured at the top of each script via a config dictionary. The selected best hyperparameters for each method are presented in Appendix B of our paper.

In case you don't want to use Weights & Biases (for logging):
```bash
export WANDB_MODE=disabled
```

1. Linear tracking
```
python3 train_linear.py
```

2. Non-linear tracking
```
python3 train_nonlinear_multi_op.py
```

3. Streaming permuted EMNIST
```
python3 train_permuted_emnist.py
```

## Dependencies

- numpy==2.2.5
- torch==2.7.0
- torchvision==0.22.0
- wandb==0.19.11

You can install the required packages using `pip`:

```bash
pip install numpy==2.2.5 torch==2.7.0 torchvision==0.22.0 wandb==0.19.11
```
