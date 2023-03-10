import pickle
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import os
import uuid
from scipy.stats import halfnorm
import math

from Auxiliar.ClassEnv import create_rllib_env


# based on https://github.com/ray-project/ray/issues/6669
class PrioritizedSelfPlay(DefaultCallbacks):

    def __init__(self, freq_save=5, freq_run=1):
        super().__init__()
        self.freq_save = freq_save
        self.freq_run = freq_run
        self.counter = 0
        self.opponent_iter = 0
        self.policy_history = []

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        self.counter += 1

        # Current result
        current_hist = {
            'iteraction': algorithm.iteration,
            'weights_filepath': None,
            'opponent_iter': self.opponent_iter,
            'result': result['policy_reward_mean']['current_team']
        }

        ## Save current checkpoint
        if self.counter % self.freq_save == 0:
            checkpoint_dir = os.path.join(algorithm.logdir, 'selfplay_checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            weights_filepath = os.path.join(checkpoint_dir, str(algorithm.iteration) + '.pkl')

            with open(weights_filepath, 'wb') as f:
                weights = algorithm.get_weights(['current_team'])['current_team']
                pickle.dump(weights, f)

            current_hist['weights_filepath'] = weights_filepath
            print('Policy Saved:', current_hist)
            self.policy_history.append(current_hist)

        # sample opponent policy
        if self.counter % self.freq_run == 0:
            # 50% of the time current policy, 50% random policy
            if np.random.uniform(0, 1) < 0.1 or len(self.policy_history) == 0:
                opponent_policy = current_hist
                opponent_weights = algorithm.get_weights(['current_team'])['current_team']
            else:
                max_random_val = len(self.policy_history)
                # random_val = max_random_val - halfnorm.rvs(scale=max_random_val/1.5, size=1)[0]
                # random_val = 0 if random_val < 0 else math.floor(random_val)
                random_val = math.floor(np.random.triangular(0, max_random_val, max_random_val))
                opponent_policy = self.policy_history[random_val]
                with open(opponent_policy['weights_filepath'], 'rb') as f:
                    opponent_weights = pickle.load(f)

        print('sampled opponent policy:', opponent_policy)
        self.opponent_iter = opponent_policy['iteraction']

        algorithm.set_weights({'current_team': algorithm.get_weights(['current_team'])['current_team'],
                             'opponent_team': opponent_weights,
                               })


# based on https://github.com/ray-project/ray/issues/7023
# Select a random team (blue or yellow) to give the current_policy or opponent_policy
class MatchMaker:
    def __init__(self, policy_ids, n_agents=4):
        self.policy_ids = policy_ids
        self.n_agents = n_agents

        self.team_select = 0  # 0 or 1
        # self.uuid = uuid.uuid4() #for debug
        self.i = 0

    def policy_mapping_fn(self, agent_id):
        self.i += 1

        if self.i <= self.n_agents // 2:
            select_policy = self.team_select
        else:
            select_policy = 1 - self.team_select

        policy_id = self.policy_ids[select_policy]

        # print('MatchMaker', agent_id, policy_id, self.i, self.uuid)

        # prepare next iteration
        if self.i == self.n_agents:
            self.team_select = np.random.choice([0, 1])  # choose a random team
            # self.uuid = uuid.uuid4() #for debug
            self.i = 0

        return policy_id


NUM_ENVS_PER_WORKER = 1

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    policies = {
        "current_team": (None, obs_space, act_space, {}),
        "opponent_team": (None, obs_space, act_space, {}),
    }

    matchmaker = MatchMaker(list(policies.keys()))

    analysis = tune.run(
        "PPO",
        name="PPO_deepmind_selfplay_v4",
        config={
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 256,
            "lr": 3e-4,
            "lambda": .95,
            "gamma": .99,
            "clip_param": 0.2,
            "num_sgd_iter": 20,
            "rollout_fragment_length": 200,
            "model": {
                "fcnet_hiddens": [1024, 512, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": matchmaker.policy_mapping_fn,
                "policies_to_train": ["current_team"]
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            'callbacks': MultiCallbacks([PrioritizedSelfPlay])
        },
        stop={
            "timesteps_total": 20000000,  # 15M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="../ray_results",
        # restore="../ray_results/PPO_deepmind_selfplay_v4/PPO_Soccer_ad2cb_00000_0_2021-12-02_23-21-09/checkpoint_001800/checkpoint-1800",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets the best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
