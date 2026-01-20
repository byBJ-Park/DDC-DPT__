import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import gymnasium as gym
from mlp import MLP
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_DATASET_FILES = {
    "LL": "D_LunarLander_medium_mixed.npz",
    "CP": "D_Cartpole_medium_mixed.npz",
    "AC": "D_Acrobot_medium_mixed.npz",
}
ENV_ALIASES = {
    "LL": ["lunarlander"],
    "CP": ["cartpole"],
    "AC": ["acrobot"],
}
ENV_IDS = {
    "LL": "LunarLander-v3",
    "CP": "CartPole-v1",
    "AC": "Acrobot-v1",
}


def resolve_npz_path(config):
    dataset_file = config.get("dataset_file")
    if dataset_file and os.path.isfile(dataset_file):
        return dataset_file

    gym_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(gym_dir, ".."))
    dataset_dir_config = config.get("dataset_dir", "datasets")
    if os.path.isabs(dataset_dir_config):
        dataset_dirs = [dataset_dir_config]
    else:
        dataset_dirs = [
            os.path.join(gym_dir, dataset_dir_config),
            os.path.join(repo_root, dataset_dir_config),
        ]

    env = config.get("env")
    filename = ENV_DATASET_FILES.get(env)
    aliases = ENV_ALIASES.get(env, [])

    for dataset_dir in dataset_dirs:
        if not os.path.isdir(dataset_dir):
            continue
        if filename:
            candidate = os.path.join(dataset_dir, filename)
            if os.path.isfile(candidate):
                return candidate
        candidates = [f for f in os.listdir(dataset_dir) if f.endswith(".npz")]
        for alias in aliases:
            for candidate in candidates:
                if alias in candidate.lower():
                    return os.path.join(dataset_dir, candidate)
    return None


class Dataset(torch.utils.data.Dataset):
    """Dataset class for storing and sampling (s, a, r, s') transitions."""

    def __init__(self, path, config):
        self.store_gpu = config.get('store_gpu', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not str(path).endswith(".npz"):
            raise ValueError(f"Expected an .npz dataset file, got: {path}")

        data = np.load(path)
        states = np.asarray(data["obs"])
        actions = np.asarray(data["act"])
        next_states = np.asarray(data["obs2"])
        rewards = np.asarray(data["rew"])
        done = np.asarray(data["done"])
        self.episode_returns = np.asarray(data.get("episode_returns", []))

        self.dataset = {
            'states': self.convert_to_tensor(states, store_gpu=self.store_gpu),
            'actions': self.convert_to_tensor(actions, store_gpu=self.store_gpu),
            'next_states': self.convert_to_tensor(next_states, store_gpu=self.store_gpu),
            'rewards': self.convert_to_tensor(rewards, store_gpu=self.store_gpu),
            'done': self.convert_to_tensor(done, store_gpu=self.store_gpu)
        }

        if config.get('shuffle', False):
            self.shuffle_dataset()

    def __len__(self):
        return len(self.dataset['states'])

    def __getitem__(self, idx):
        return {
            'states': self.dataset['states'][idx],
            'actions': self.dataset['actions'][idx],
            'next_states': self.dataset['next_states'][idx],
            'rewards': self.dataset['rewards'][idx],
            'done': self.dataset['done'][idx]
        }

    def shuffle_dataset(self):
        indices = np.arange(len(self.dataset['states']))
        np.random.shuffle(indices)
        for key in self.dataset.keys():
            self.dataset[key] = self.dataset[key][indices]

    @staticmethod
    def convert_to_tensor(x, store_gpu):
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        return tensor.to("cuda") if store_gpu else tensor


def build_log_filename(config):
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_lr{config['lr']}"
                f"_batch{config['batch_size']}"
                f"_decay{config['decay']}"
                f"_clip{config['clip']}")
    filename += f'_{timestamp}'
    return filename + ".log"


def printw(message, config):
    print(message)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = build_log_filename(config)
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "a") as log_file:
        print(message, file=log_file)


def evaluate_policy(model, env_id, episodes, device):
    """Run online evaluation and return average return."""
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
    return float(np.mean(returns)) if returns else None


def plot_online_eval(online_eval_steps, online_eval_returns, config, env_id):
    """Plot online evaluation results over gradient updates."""
    if not online_eval_steps or not online_eval_returns:
        return

    os.makedirs("figs/eval", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(online_eval_steps, online_eval_returns, label="GLADIUS", color='red', linewidth=2)
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.title(f"GLADIUS Performance on {env_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/eval/{build_log_filename(config)}_online_eval.png")
    plt.close()


def plot_training_losses(train_be_loss, train_D_loss, test_r_MAPE_loss, eval_steps, config, rep):
    """Plot training losses with zero-value filtering."""
    plt.figure(figsize=(12, 9))

    # BE Loss
    plt.subplot(3, 1, 1)
    plt.yscale('log')
    plt.xlabel('Update Step')
    plt.ylabel('Train BE Loss')
    if train_be_loss:
        non_zero = [(i, v) for i, v in enumerate(train_be_loss) if v > 0]
        if non_zero:
            steps, values = zip(*non_zero)
            plt.plot(steps, values, label="Bellman Error Loss", color='red')
    plt.legend()

    # D Loss
    plt.subplot(3, 1, 2)
    plt.yscale('log')
    plt.xlabel('Update Step')
    plt.ylabel('D Loss')
    if train_D_loss:
        non_zero = [(i, v) for i, v in enumerate(train_D_loss) if v > 0]
        if non_zero:
            steps, values = zip(*non_zero)
            plt.plot(steps, values, label="D Loss", color='orange')
    plt.legend()

    # R MAPE Loss
    plt.subplot(3, 1, 3)
    plt.yscale('log')
    plt.xlabel('Update Step')
    plt.ylabel('Test R MAPE Loss')
    if test_r_MAPE_loss:
        non_zero = [(i, v) for i, v in zip(eval_steps[:len(test_r_MAPE_loss)], test_r_MAPE_loss) if v > 0]
        if non_zero:
            steps, values = zip(*non_zero)
            plt.plot(steps, values, label="R MAPE Loss", color='purple')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"figs/loss/{build_log_filename(config)}_rep{rep}_losses.png")
    plt.close()


def train(config):
    os.makedirs('figs/loss', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Set random seeds
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Prepare dataset
    dataset_config = {
        'num_trajs': config['num_trajs'],
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env']
    }

    npz_path = resolve_npz_path(config)
    if not npz_path:
        raise FileNotFoundError(
            "Offline dataset not found. Expected dataset_dir to contain "
            f"{ENV_DATASET_FILES.get(config.get('env'))} or pass dataset_file."
        )
    dataset_config['dataset_file'] = npz_path
    train_dataset = Dataset(npz_path, dataset_config)
    test_dataset = train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )

    # Environment config
    if config['env'] == 'LL':
        states_dim, actions_dim, init_b_value = 8, 4, 100
    elif config['env'] == 'CP':
        states_dim, actions_dim, init_b_value = 4, 2, 1
    elif config['env'] == 'AC':
        states_dim, actions_dim, init_b_value = 6, 3, -10
    else:
        print('Invalid environment')
        exit()

    def custom_output_b_init(bias):
        nn.init.constant_(bias, init_b_value)

    # Prepare model
    model_config = {
        'hidden_sizes': [config['h_size']] * config['n_layer'],
        'layer_normalization': config['layer_norm'],
    }
    model = MLP(states_dim, actions_dim, output_b_init=custom_output_b_init, **model_config).to(device)

    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    MSE_loss_fn = torch.nn.MSELoss(reduction='mean')
    MAE_loss_fn = torch.nn.L1Loss(reduction='mean')

    repetitions = config['repetitions']

    episode_returns = np.asarray(getattr(train_dataset, "episode_returns", []))
    if episode_returns.size > 0:
        printw(
            f"Episode returns summary - count={episode_returns.size}, "
            f"mean={episode_returns.mean():.4f}, std={episode_returns.std():.4f}, "
            f"min={episode_returns.min():.4f}, max={episode_returns.max():.4f}",
            config,
        )

    rep_test_r_MAPE_loss = []
    rep_eval_steps = None
    rep_best_r_MAPE_loss = []
    rep_episode_returns = []

    # Online evaluation 설정
    env_id = ENV_IDS.get(config.get("env"))
    online_eval_interval = config.get('online_eval_interval', 1000)
    online_eval_episodes = config.get('online_eval_episodes', 10)

    for rep in range(repetitions):
        print(f"\nStarting repetition {rep+1}/{repetitions}")
        if episode_returns.size > 0:
            rep_episode_returns.append(episode_returns)

        train_be_loss = []
        train_D_loss = []
        test_r_MAPE_loss = []
        eval_steps = []
        online_eval_steps = []
        online_eval_returns = []
        best_update = -1
        best_r_MAPE_loss = 9999

        num_updates = int(config['num_epochs'])
        eval_interval = max(1, int(config.get('eval_interval', len(train_loader))))
        train_iter = iter(train_loader)

        # D:Q ratio 설정 (기본값 1 = D,Q,D,Q,...)
        d_q_ratio = config.get('d_q_ratio', 1)
        update_cycle = d_q_ratio + 1

        for update_step in tqdm(range(num_updates), desc="Training Progress"):

            ### ONLINE EVALUATION ###
            if env_id and online_eval_episodes > 0:
                if (update_step + 1) % online_eval_interval == 0 or update_step == 0:
                    avg_return = evaluate_policy(model, env_id, online_eval_episodes, device)
                    online_eval_steps.append(update_step + 1)
                    online_eval_returns.append(avg_return)
                    printw(f"Online eval at step {update_step + 1}: avg_return={avg_return:.2f}", config)

            ### EVALUATION ###
            if (update_step + 1) % eval_interval == 0 or update_step == 0:
                printw(f"Update: {update_step + 1}", config)
                start_time = time.time()
                with torch.no_grad():
                    epoch_r_MAPE_loss = 0.0
                    for i, batch in enumerate(test_loader):
                        print(f"Batch {i} of {len(test_loader)}", end='\r')
                        batch = {k: v.to(device) for k, v in batch.items()}
                        pred_q_values, _, pred_vnext_values = model(batch)
                        true_actions = batch['actions'].long()

                        pred_r_values = pred_q_values - config['beta'] * pred_vnext_values
                        chosen_pred_r_values = torch.gather(pred_r_values, dim=1, index=true_actions.unsqueeze(-1))
                        true_r_values = batch['rewards']

                        diff = torch.abs(chosen_pred_r_values - true_r_values)
                        r_MAPE = torch.mean(diff)
                        epoch_r_MAPE_loss += r_MAPE.item()

                    if epoch_r_MAPE_loss / len(test_loader) < best_r_MAPE_loss:
                        best_r_MAPE_loss = epoch_r_MAPE_loss / len(test_loader)
                        best_update = update_step

                test_r_MAPE_loss.append(epoch_r_MAPE_loss / len(test_loader))
                eval_steps.append(update_step + 1)
                end_time = time.time()
                printw(f"\tMAPE of r(s,a): {test_r_MAPE_loss[-1]}", config)
                printw(f"\tEval time: {end_time - start_time}", config)

            ### TRAINING ###
            epoch_train_be_loss = 0.0
            epoch_train_D_loss = 0.0
            start_time = time.time()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            pred_q_values, pred_q_values_next, pred_vnext_values = model(batch)
            true_actions = batch['actions'].long()
            true_rewards = batch['rewards']

            chosen_q_values = torch.gather(pred_q_values, dim=1, index=true_actions.unsqueeze(-1))
            chosen_vnext_values = torch.gather(pred_vnext_values, dim=1, index=true_actions.unsqueeze(-1))

            logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1)
            vnext = logsumexp_nextstate
            done = batch['done'].to(torch.bool)
            vnext = torch.where(done, torch.tensor(0.0, device=vnext.device), vnext)

            is_d_step = (update_step % update_cycle) < d_q_ratio

            if is_d_step:  # D update
                D = MSE_loss_fn(vnext.clone().detach(), chosen_vnext_values)
                D.backward()

                current_lr_vnext = config['lr'] / (1 + config['decay'] * update_step)
                vnext_optimizer.param_groups[0]['lr'] = current_lr_vnext

                if config['clip']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])

                vnext_optimizer.step()
                vnext_optimizer.zero_grad()
                epoch_train_D_loss += D.item()
                model.zero_grad()

            else:  # Q update
                pivot_rewards = true_rewards
                td_error = chosen_q_values - pivot_rewards - config['beta'] * vnext
                td_error = torch.where(done, chosen_q_values - pivot_rewards, td_error)

                vnext_dev = vnext - chosen_vnext_values.clone().detach()
                # Deterministic environment에서는 bias correction term을 0으로 설정
                if config.get("zero_bias_correction", False):
                    vnext_dev = torch.zeros_like(vnext_dev)

                be_error = td_error ** 2 - config['beta'] ** 2 * vnext_dev ** 2
                be_loss = MAE_loss_fn(be_error, torch.zeros_like(be_error))
                be_loss.backward()

                current_lr_q = config['lr'] / (1 + config['decay'] * update_step)
                q_optimizer.param_groups[0]['lr'] = current_lr_q

                if config['clip']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])

                q_optimizer.step()
                q_optimizer.zero_grad()
                model.zero_grad()
                epoch_train_be_loss += be_loss.item()

            # Log initial predictions
            if update_step == 0:
                pred_r_values_print = pred_q_values[:10, :] - config['beta'] * pred_vnext_values[:10, :]
                chosen_r_values_print = torch.gather(pred_r_values_print, dim=1, index=true_actions[:10].unsqueeze(-1))
                true_r_values_print = true_rewards[:10].unsqueeze(1)
                actions_print = true_actions[:10].int().unsqueeze(1)
                pred_r_values_with_true_r = torch.cat((actions_print, true_r_values_print, chosen_r_values_print), dim=1)
                pred_r_values_np = pred_r_values_with_true_r.cpu().clone().detach().numpy()
                np.set_printoptions(suppress=True, precision=6)
                printw(f"Predicted r values: {pred_r_values_np}", config)

            train_be_loss.append(epoch_train_be_loss)
            train_D_loss.append(epoch_train_D_loss)

            end_time = time.time()
            printw(f"\tBE loss: {train_be_loss[-1]}", config)
            printw(f"\tTrain time: {end_time - start_time}", config)

            # Save model periodically
            if (update_step + 1) % 1000 == 0:
                torch.save(model.state_dict(),
                           f'models/{build_log_filename(config)}_rep{rep}_update{update_step+1}.pt')
                plot_training_losses(train_be_loss, train_D_loss, test_r_MAPE_loss, eval_steps, config, rep)

        # End of repetition
        printw(f"Best update for repetition {rep+1}: {best_update}", config)
        printw(f"Best R MAPE loss for repetition {rep+1}: {best_r_MAPE_loss}", config)

        if best_update > 0:
            rep_best_r_MAPE_loss.append(best_r_MAPE_loss)
        else:
            printw("No best r values were recorded during training.", config)

        rep_test_r_MAPE_loss.append(test_r_MAPE_loss)
        if rep_eval_steps is None:
            rep_eval_steps = eval_steps

        torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')

        # Plot online evaluation (Gradient Updates 기준)
        if online_eval_steps and online_eval_returns:
            plot_online_eval(online_eval_steps, online_eval_returns, config, env_id)

        printw(f"\nTraining of repetition {rep+1} finished.", config)

    # Final results
    rep_test_r_MAPE_loss = np.array(rep_test_r_MAPE_loss)
    mean_r_mape = np.mean(rep_test_r_MAPE_loss, axis=0)
    std_r_mape = np.std(rep_test_r_MAPE_loss, axis=0) / np.sqrt(repetitions)
    eval_steps = np.asarray(rep_eval_steps) if rep_eval_steps else np.arange(len(mean_r_mape))

    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    plt.xlabel('Update')
    plt.ylabel('R MAPE Loss')
    plt.plot(eval_steps, mean_r_mape, label="Mean R MAPE Loss", color='blue')
    plt.fill_between(eval_steps, mean_r_mape - std_r_mape, mean_r_mape + std_r_mape, alpha=0.2, color='blue')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/loss/Reps{repetitions}_{build_log_filename(config)}_losses.png")
    plt.close()

    printw(f"\nTraining completed.", config)
    mean_best_r_mape = np.mean(rep_best_r_MAPE_loss)
    std_best_r_mape = np.std(rep_best_r_MAPE_loss) / np.sqrt(repetitions)
    printw(f"\nFinal results for {repetitions} repetitions", config)
    printw(f"Mean best R MAPE loss: {mean_best_r_mape}", config)
    printw(f"Standard error of best R MAPE loss: {std_best_r_mape}", config)
