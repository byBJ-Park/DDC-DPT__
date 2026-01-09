import gymnasium as gym
from pyvirtualdisplay import Display
from stable_baselines3 import PPO
import warnings
import torch
import torch.nn
from torch import nn
from torch.nn import functional as F
from multiHeadedMLPModule import MultiHeadedMLPModule
import pandas as pd


class MLP(MultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 states_dim,
                 actions_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=lambda x: nn.init.constant_(x, 15),
                 #output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(2, states_dim, [actions_dim, actions_dim], hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization) #fix n_heads to 2 (q and v heads) with 2 output dimensions each (actions_dim)
    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            Return Q-values and E[V(s',a')|s,a]-values.

        """
        states = x #dimension is (batch_size, state_dim)
        
        q_values, vnext_values = super().forward(states) #dim is (batch_size*horizon, action_dim)
        
        #bound all outputs to non-negative values, using softplus
        q_values = F.softplus(q_values)
        vnext_values = F.softplus(vnext_values)
        
        return q_values, vnext_values



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# When loading the model, specify the custom objects
custom_objects = {
    "clip_range": 1,  # This is the default PPO clip range
    "learning_rate": 0.0003  # Default PPO learning rate
}


config = {
    'n_layer': 2,
    'h_size': 64,
    'layer_norm': False,
    'env': 'LL',
    'setup': 'IRL_RL',
    'episodes': 100
}

display = Display(visible=0, size=(1400, 900))
display.start()


if config['env'] == 'LL':
    states_dim = 8
    actions_dim = 4
    env = gym.make("LunarLander-v2")
    model_path = 'models/LL_num_trajs15_lr0.0005_batch128_decay0.001_clip1_20250207.log_rep0_epoch5000.pt' 
    IQ_path = 'models/IQ_LL_num_trajs15_lr0.001_batch64_decay0.0001_clipFalse_20250128.log_rep0_epoch1000.pt'
    expert_path = "Expert_policy/LunarLander-v2_PPO.zip"
    IRL_RL_path = "PPO_IRL/PPO-15-LL.zip"
    expert_reward = 232.77 
    
elif config['env'] == 'AC': #Acrobot
    states_dim = 6
    actions_dim = 3
    env = gym.make("Acrobot-v1")
    model_path = 'models/AC_num_trajs3_lr0.0005_batch128_decay0.001_clip1_20250130.log_rep0_epoch8000.pt'
    IQ_path = 'models/IQ_AC_num_trajs7_lr0.001_batch64_decay0.0001_clipFalse_20250128.log_rep0_epoch4000.pt'
    expert_path = "Expert_policy/Acrobot-v1_PPO.zip"
    IRL_RL_path = "PPO_IRL/PPO-AC.zip"
    expert_reward = -82.80
    
elif config['env'] == 'CP': #CartPole
    states_dim = 4
    actions_dim = 2
    env = gym.make("CartPole-v1")
    model_path = 'dd'
    IQ_path = 'models/IQ_CP_num_trajs7_lr0.001_batch64_decay0.0001_clipFalse_20250128.log_rep0_epoch3000.pt'
    expert_path = "Expert_policy/CartPole-v1_PPO.zip"
    IRL_RL_path = "PPO_IRL/PPO-CP.zip"
    expert_reward = 500
    
model_config = {
        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], 
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['setup'] == 'our':
    model = MLP(states_dim, actions_dim, **model_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
elif config['setup'] == 'IQ':
    model = MLP(states_dim, actions_dim, **model_config).to(device)
    model.load_state_dict(torch.load(IQ_path))
    model.to(device)
    model.eval()
elif config['setup'] == 'expert':
    model = PPO.load(expert_path, custom_objects=custom_objects)
elif config['setup'] == 'IRL_RL':
    model = PPO.load(IRL_RL_path, custom_objects=custom_objects)


# Create the environment


# Evaluation loop without rendering
results = []
episodes = config['episodes']

for episode in range(config['episodes']):
    obs, _ = env.reset()

    total_reward = 0
    done = False
    
    while not done:
        if config['setup'] == 'our' or config['setup'] == 'IQ':
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                obs = obs.unsqueeze(0)
            
                q_predict, _ = model(obs)
                #softmax action choice in terms of q
                action_prob = F.softmax(q_predict)
                chosen_action = torch.multinomial(action_prob, 1).item()            
        else: #expert
            chosen_action, _states = model.predict(obs)
            if config['env'] == 'LL':   
                if (obs[6] ==1) & (obs[7] ==1):
                    chosen_action = 0 #A true expert will do this action
        obs, reward, terminated, truncated, _ = env.step(chosen_action)
        done = terminated or truncated
        total_reward += reward
    
    results.append({'Episode': episode + 1, 'Total Reward': total_reward/expert_reward})
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

df = pd.DataFrame(results)

if config['setup'] == 'our':
    df.to_csv(f'csvs/our_{config["env"]}_itr{episodes}_3.csv', index=False)
elif config['setup'] == 'IQ':
    df.to_csv(f'csvs/IQ_{config["env"]}_itr{episodes}.csv', index=False)
elif config['setup'] == 'expert':
    df.to_csv(f'csvs/expert_{config["env"]}_itr{episodes}.csv', index=False)
elif config['setup'] == 'IRL_RL':
    df.to_csv(f'csvs/IRL_RL_{config["env"]}_itr{episodes}.csv', index=False)
else:
    print('Invalid setup')
    exit()