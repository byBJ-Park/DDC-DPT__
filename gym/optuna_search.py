"""Optuna hyperparameter search for gym training."""

import os
import sys
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import optuna
from optuna.trial import TrialState
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime

from gym_train import (
    DEVICE, ENV_IDS, ENV_CONFIGS, Dataset, resolve_npz_path,
    set_random_seeds, ensure_directories, MLP, nn
)

# =============================================================================
# Weight Initialization Functions
# =============================================================================

WEIGHT_INIT_FNS = {
    "xavier": nn.init.xavier_normal_,
    "orthogonal": lambda w: nn.init.orthogonal_(w, gain=np.sqrt(2)),
    "kaiming": lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
}

# =============================================================================
# Objective Function
# =============================================================================


def evaluate_policy_silent(model, env_id, episodes, device):
    """Evaluate policy without logging."""
    env = gym.make(env_id)
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            batch = {'states': obs_tensor, 'next_states': obs_tensor}

            with torch.no_grad():
                q_values, _, _ = model(batch)
                action_probs = torch.softmax(q_values, dim=1)
                action = torch.multinomial(action_probs, num_samples=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

    env.close()
    return float(np.mean(returns))


def create_objective(base_config, search_config):
    """Create objective function with base config."""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""

        # =============================================================
        # Hyperparameter suggestions
        # =============================================================

        # 핵심 파라미터
        beta = trial.suggest_categorical("beta", [0.9, 0.95, 0.99])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        decay = trial.suggest_float("decay", 1e-6, 1e-3, log=True)

        # 네트워크 구조
        h_size = trial.suggest_categorical("h_size", [64, 128, 256])
        n_layer = trial.suggest_int("n_layer", 2, 4)
        layer_norm = trial.suggest_categorical("layer_norm", [True, False])

        # Weight initialization
        weight_init = trial.suggest_categorical("weight_init", ["xavier", "orthogonal", "kaiming"])

        # 학습 설정
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        reward_scale = trial.suggest_categorical("reward_scale", [1, 5, 10, 20, 100])
        clip = trial.suggest_categorical("clip", [1.0, 5.0, 10.0])

        # D:Q 업데이트 비율 (D를 몇 번 업데이트 후 Q를 1번 업데이트)
        d_q_ratio = trial.suggest_categorical("d_q_ratio", [1, 2, 3, 5])

        # =============================================================
        # Build config (merge with base config)
        # =============================================================

        config = {
            **base_config,
            "beta": beta,
            "lr": lr,
            "decay": decay,
            "h_size": h_size,
            "n_layer": n_layer,
            "layer_norm": layer_norm,
            "weight_init": weight_init,
            "batch_size": batch_size,
            "reward_scale": reward_scale,
            "clip": clip,
            "d_q_ratio": d_q_ratio,
        }

        # =============================================================
        # Setup
        # =============================================================

        set_random_seeds(config['seed'])

        # Dataset
        npz_path = resolve_npz_path(config)
        if not npz_path:
            raise FileNotFoundError("Dataset not found")

        train_dataset = Dataset(npz_path, config)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )

        # Model
        env_config = ENV_CONFIGS[config['env']]
        states_dim = env_config['states_dim']
        actions_dim = env_config['actions_dim']
        init_b_value = env_config['init_b_value']

        def custom_output_b_init(bias):
            nn.init.constant_(bias, init_b_value)

        # Get weight init function
        weight_init_fn = WEIGHT_INIT_FNS[weight_init]

        model = MLP(
            states_dim, actions_dim,
            hidden_w_init=weight_init_fn,
            output_w_init=weight_init_fn,
            output_b_init=custom_output_b_init,
            hidden_sizes=[config['h_size']] * config['n_layer'],
            layer_normalization=config['layer_norm']
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)

        # =============================================================
        # Training loop with pruning
        # =============================================================

        env_id = ENV_IDS[config['env']]
        num_updates = search_config['num_updates']
        eval_interval = search_config['eval_interval']
        eval_episodes = search_config['eval_episodes']

        train_iter = iter(train_loader)
        best_return = float('-inf')

        # D:Q ratio를 위한 카운터
        d_q_ratio = config['d_q_ratio']
        update_cycle = d_q_ratio + 1  # D를 d_q_ratio번, Q를 1번

        for update_step in range(num_updates):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            pred_q_values, pred_q_values_next, pred_vnext_values = model(batch)
            true_actions = batch['actions'].long()
            true_rewards = batch['rewards']
            done = batch['done'].to(torch.bool)

            # D:Q 비율에 따른 업데이트
            # 예: d_q_ratio=2이면 [D, D, Q, D, D, Q, ...]
            cycle_pos = update_step % update_cycle
            is_d_step = cycle_pos < d_q_ratio

            if is_d_step:
                # D step
                chosen_vnext_values = torch.gather(pred_vnext_values, dim=1, index=true_actions.unsqueeze(-1))
                logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1)
                vnext = torch.where(done, torch.tensor(0.0, device=DEVICE), logsumexp_nextstate)

                loss = torch.nn.MSELoss()(vnext.detach(), chosen_vnext_values.squeeze(-1))
            else:
                # Q step
                chosen_q_values = torch.gather(pred_q_values, dim=1, index=true_actions.unsqueeze(-1))
                chosen_vnext_values = torch.gather(pred_vnext_values, dim=1, index=true_actions.unsqueeze(-1))
                logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1)
                vnext = torch.where(done, torch.tensor(0.0, device=DEVICE), logsumexp_nextstate)

                td_error = chosen_q_values - true_rewards - config['beta'] * vnext
                td_error = torch.where(done, chosen_q_values - true_rewards, td_error)

                vnext_dev = vnext - chosen_vnext_values.clone().detach()
                if config.get("zero_bias_correction", False):
                    vnext_dev = torch.zeros_like(vnext_dev)

                be_error = td_error**2 - config['beta']**2 * vnext_dev**2
                loss = torch.nn.L1Loss()(be_error, torch.zeros_like(be_error))

            loss.backward()

            # LR decay
            current_lr = config['lr'] / (1 + config['decay'] * update_step)
            optimizer.param_groups[0]['lr'] = current_lr

            if config['clip']:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])

            optimizer.step()
            optimizer.zero_grad()

            # =============================================================
            # Evaluation & Pruning
            # =============================================================

            if (update_step + 1) % eval_interval == 0:
                avg_return = evaluate_policy_silent(model, env_id, eval_episodes, DEVICE)

                if avg_return > best_return:
                    best_return = avg_return

                # Report to Optuna for pruning
                trial.report(avg_return, update_step)

                # Prune if not promising
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Final evaluation
        final_return = evaluate_policy_silent(model, env_id, eval_episodes, DEVICE)

        return max(best_return, final_return)

    return objective


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--config", type=str, required=True, help="Path to base config JSON file")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--num_updates", type=int, default=30000, help="Updates per trial")
    parser.add_argument("--eval_interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r") as f:
        full_config = json.load(f)

    global_config = full_config.get('global_config', {})
    training_config = full_config.get('training_config', {})
    env_config = full_config.get('env_config', {})
    experiments = full_config.get('experiments', [{}])

    # Merge base config
    base_config = {**global_config, **training_config, **env_config, **experiments[0]}

    # Search config
    search_config = {
        "n_trials": args.n_trials,
        "num_updates": args.num_updates,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
    }

    study_name = args.study_name or f"{base_config.get('env', 'unknown')}_search"

    ensure_directories("optuna_results", "configs")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10000,
            interval_steps=5000,
        ),
        storage=f"sqlite:///optuna_results/{study_name}.db",
        load_if_exists=True,
    )

    print("=" * 60)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Environment: {base_config.get('env')}")
    print(f"Trials: {args.n_trials}")
    print(f"Updates per trial: {args.num_updates}")
    print(f"Study name: {study_name}")
    print(f"Database: optuna_results/{study_name}.db")
    print("=" * 60 + "\n")

    # Run optimization
    study.optimize(
        create_objective(base_config, search_config),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # =================================================================
    # Results
    # =================================================================

    print("\n" + "=" * 60)
    print("SEARCH COMPLETED")
    print("=" * 60)

    # Best trial
    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Return: {best_trial.value:.2f}")
    print("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Statistics
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print(f"\nStatistics:")
    print(f"  Completed trials: {len(complete_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")

    # Save best config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_config = {
        "global_config": {
            "beta": best_trial.params["beta"]
        },
        "training_config": {
            "h_size": best_trial.params["h_size"],
            "n_layer": best_trial.params["n_layer"],
            "shuffle": True,
            "seed": base_config.get("seed", 1),
            "store_gpu": True,
            "layer_norm": best_trial.params["layer_norm"],
            "weight_init": best_trial.params["weight_init"],
            "num_updates": 100000,
            "repetitions": 1,
            "zero_bias_correction": base_config.get("zero_bias_correction", True),
            "online_eval_interval": 1000,
            "online_eval_episodes": 10,
            "lr": best_trial.params["lr"],
            "decay": best_trial.params["decay"],
            "clip": best_trial.params["clip"],
            "d_q_ratio": best_trial.params["d_q_ratio"]
        },
        "env_config": {
            "env": base_config.get("env", "LL")
        },
        "experiments": [
            {
                "batch_size": best_trial.params["batch_size"],
                "reward_scale": best_trial.params["reward_scale"]
            }
        ]
    }

    config_path = f"configs/gym_best_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to: {config_path}")

    # Top 5 trials
    if complete_trials:
        print("\n" + "-" * 60)
        print("Top 5 Trials:")
        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)[:5]
        for i, t in enumerate(sorted_trials, 1):
            print(f"  {i}. Trial #{t.number}: {t.value:.2f}")
            print(f"     beta={t.params['beta']}, lr={t.params['lr']:.6f}, "
                  f"weight_init={t.params['weight_init']}, d_q_ratio={t.params['d_q_ratio']}")


if __name__ == "__main__":
    main()
