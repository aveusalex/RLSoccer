# from time import time
from collections import defaultdict
from tqdm import tqdm
# import sys
import random
import numpy as np
import torch
# from torch import nn
import plotly.express as px
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from soccer_twos import EnvType
# import gymnasium as gym
from ray import tune
# from ray.rllib import MultiAgentEnv
# import soccer_twos
from ray.tune.logger import pretty_print
from Auxiliar.ClassEnv import create_rllib_env
import ray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ppo_config = PPOConfig()

tune.registry.register_env("Soccer", create_rllib_env)  # registrando o ambiente no tune

environment_id = "Soccer"

ppo_config = ppo_config.resources(
    num_gpus = 1,
    num_cpus_per_worker = 0,
)

ppo_config = ppo_config.rollouts(
    num_rollout_workers = 4,
    # Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the local worker (driver process or the Algorithm’s actor when using Tune).
    num_envs_per_worker = 1,
    # Number of environments to evaluate vector-wise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
    rollout_fragment_length = 8,
    # Divide episodes into fragments of this many steps each during rollouts.
)

ppo_config = ppo_config.environment(
    env = environment_id,
)

ppo_config.env_config = {"render": False, "time_scale": 80, "multiagent": False, "variation": EnvType.team_vs_policy,
                         "flatten_branched": True, "single_player": True}  # colocando os parâmertros do ambiente (soccer-twos)

ppo_config = ppo_config.framework(
    framework = "torch",
)

ppo_config = ppo_config.training(
    lr = 5e-2,
    #  The default learning rate.
    train_batch_size = 128, # deve ser múltiplo de (workers * env_per_worker * rollout_fragment_length)
    # Training batch size, if applicable.
    use_critic = True,
    # Should use a critic as a baseline (otherwise don’t use value baseline; required for using GAE).
    use_gae = True,
    # If true, use the Generalized Advantage Estimator (GAE) with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    gamma = 0.99,
    # Float specifying the discount factor of the Markov Decision process.
    lambda_ = 0.95,
    # The GAE (lambda) parameter.
    sgd_minibatch_size = 32,
    # Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
    num_sgd_iter = 6,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
    shuffle_sequences = True,
    # Whether to shuffle sequences in the batch when training (recommended).
    vf_loss_coeff = 0.5,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model’s config.
    entropy_coeff = 0.0,
    # Coefficient of the entropy regularizer.
    vf_clip_param = 100000.0, #Aqui eliminamos o clip colocando ele muito alto
    # Clip param for the value function. Note that this is sensitive to the scale of the rewards. If your expected V is large, increase this.
    clip_param = 0.5,
    #  PPO clip parameter.
    kl_coeff = 0.0,
    # Initial coefficient for KL divergence.
    model = {
        "fcnet_hiddens": [64, 32],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },
    # Arguments passed into the policy model. See models/catalog.py for a full list of the available model options.
)

ppo_config = ppo_config.reporting(
    min_sample_timesteps_per_iteration = 1,
    metrics_num_episodes_for_smoothing = 50,
)

# PPOalgo = ppo_config.build()
# result = PPOalgo.train()
# print(pretty_print(result))
# ray.shutdown()

MAX_EPISODES = 1000

all_metrics = pd.DataFrame()
all_metrics["episodes"] = [i+1 for i in range(MAX_EPISODES)]
all_metrics["threshold_reward"] = [475 for i in range(MAX_EPISODES)]


if __name__ == '__main__':
    from collections import deque

    def run_experiment(name, config):
        PPOalgo = config.build()

        metrics = defaultdict(list)

        rew_deque = deque(maxlen=50)
        len_deque = deque(maxlen=50)

        pbar = tqdm(total=MAX_EPISODES, position=0, leave=True)

        episode = 0
        while episode < MAX_EPISODES:
            result = PPOalgo.train()

            # O código abaixo é feito para adquirir métricas com vários episódios coletados e terminados na chamada de 'train()'
            # ele serve como um exemplo do uso da métricas na variável 'result'
            if result["episodes_total"] > episode:
                for v in result["hist_stats"]["episode_reward"][-result["sampler_results"]["episodes_this_iter"]:]:
                    rew_deque.append(v)
                    metrics["train_reward"].append(np.array(rew_deque).mean())

                for v in result["hist_stats"]["episode_lengths"][-result["sampler_results"]["episodes_this_iter"]:]:
                    len_deque.append(v)
                    metrics["ep_len"].append(np.array(len_deque).mean())

                pbar.update(result["episodes_total"] - episode)
                pbar.set_description("| Mean Reward %.2f | Ep len %.2f |" % (
                result["sampler_results"]["episode_reward_mean"], result["sampler_results"]["episode_len_mean"]))

                episode = result["episodes_total"]

                # salvando o modelo a cada 10 episódios
                if episode % 10 == 0:
                    checkpoint = PPOalgo.save()
                    print("checkpoint saved at", checkpoint)

        all_metrics[name + "_reward"] = metrics["train_reward"][:1000]
        all_metrics[name + "_len"] = metrics["ep_len"][:1000]

        # !!!!! Esta execução dura ~5min
        return PPOalgo


    agent = run_experiment(name="PPO_01", config=ppo_config)

    sns.lineplot(data=all_metrics, x="episodes", y="PPO_01_reward")
    plt.show()

    sns.lineplot(data=all_metrics, x="episodes", y="PPO_01_len")
    plt.show()

    ray.shutdown()

