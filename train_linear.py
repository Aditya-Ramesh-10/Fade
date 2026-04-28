import wandb
import numpy as np
from data_generators import PeriodicLinearParameterFlip
from algs import IDBD, LMS, FADE, FADE_IDBD 
from algs import IDBD_FixedWD, LMS_FixedWD, CoupledWDSSAdaptation

PROJECT_NAME = "fade-adaptive-weight-decay"

config_dict = dict(
    seed=45,
    algorithm="fade",  # "lms", "lms_fixed_wd", "idbd", "fade", "fade_idbd", "idbd_fixed_wd", "coupled_wd_ss_adaptation"

    # problem specification
    period=20,
    num_total_components=20,
    num_flipping_components=5,
    noise_level=0.0,

    # training
    num_samples=200000,
    step_size=0.1,
    initial_beta=-4.6,
    meta_step_size=0.01,
    initial_gamma=-1.2,

    # logging
    error_over_last=200000,
    log_interval=1000,

    # used for fixed weight decay algorithms
    weight_decay=0.0001,
)


def build_algorithm(config, n):
    name = config.algorithm
    initial_beta = config.initial_beta if config.initial_beta is not None else np.log(1.0 / n)

    if name == "lms":
        return LMS(num_features=n, step_size=config.step_size, seed=config.seed)
    elif name == "lms_fixed_wd":
        return LMS_FixedWD(num_features=n, step_size=config.step_size,
                           weight_decay=config.weight_decay, seed=config.seed)
    elif name == "idbd":
        return IDBD(num_features=n, meta_step_size=config.meta_step_size,
                     initial_beta=initial_beta, seed=config.seed)
    elif name == "fade":
        return FADE(num_features=n, lr=config.step_size,
                     meta_step_size=config.meta_step_size,
                     initial_gamma=config.initial_gamma, seed=config.seed)
    elif name == "fade_idbd":
        # For simplicity, we use the same meta step size for both alpha and lambda.
        return FADE_IDBD(num_features=n,
                         meta_step_size_alpha=config.meta_step_size,
                         meta_step_size_lambda=config.meta_step_size,
                         initial_beta=initial_beta,
                         initial_gamma=config.initial_gamma,
                         seed=config.seed)
    elif name == "idbd_fixed_wd":
        return IDBD_FixedWD(num_features=n, meta_step_size=config.meta_step_size,
                            initial_beta=initial_beta, weight_decay=config.weight_decay,
                            seed=config.seed)
    elif name == "coupled_wd_ss_adaptation":
        return CoupledWDSSAdaptation(num_features=n,
                                     meta_step_size_alpha=config.meta_step_size,
                                     meta_step_size_lambda=config.meta_step_size,
                                     initial_beta=initial_beta,
                                     initial_gamma=config.initial_gamma,
                                     seed=config.seed)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def main():
    wandb.init(project=PROJECT_NAME, config=config_dict)
    config = wandb.config
    n = config.num_total_components
    n_flip = config.num_flipping_components
    name = config.algorithm

    # data generator
    online_sl_task = PeriodicLinearParameterFlip(
        period=config.period,
        num_total_components=n,
        num_flipping_components=n_flip,
        noise_level=config.noise_level,
        seed=config.seed,
    )

    alg = build_algorithm(config, n)

    error_buffer = []
    error_history = []

    for step in range(config.num_samples):
        x, y = online_sl_task.sample_training_example()
        sq_err = alg.update_weights(x, y)
        error_buffer.append(sq_err)
        error_history.append(sq_err)

        if (step + 1) % config.log_interval == 0:
            log_dict = {"step": step + 1, "mse": np.mean(error_buffer)}

            if name == "idbd" or name == "idbd_fixed_wd" or name == "fade_idbd":
                alphas = np.exp(alg.beta)
                log_dict["alpha_mean_relevant"] = np.mean(alphas[:n_flip])
                log_dict["alpha_mean_irrelevant"] = np.mean(alphas[n_flip:])

            if name == "fade" or name == "fade_idbd":
                lambdas = np.exp(alg.gamma)
                log_dict["lambda_mean_relevant"] = np.mean(lambdas[:n_flip])
                log_dict["lambda_mean_irrelevant"] = np.mean(lambdas[n_flip:])
                log_dict["lambda_max_relevant"] = np.max(lambdas[:n_flip])
                log_dict["lambda_max_irrelevant"] = np.max(lambdas[n_flip:])
                log_dict["lambda_min_relevant"] = np.min(lambdas[:n_flip])
                log_dict["lambda_min_irrelevant"] = np.min(lambdas[n_flip:])

            wandb.log(log_dict)
            error_buffer = []

    # final summary
    tail = config.error_over_last
    asymptotic_mse = np.mean(error_history[-tail:])
    wandb.summary["asymptotic_mse"] = asymptotic_mse

    wandb.finish()
    print(f"Done. {name} asymptotic MSE: {asymptotic_mse:.6f}")


if __name__ == "__main__":
    main()