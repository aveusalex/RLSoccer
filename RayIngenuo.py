from ray.rllib.algorithms.ppo import PPOConfig
from soccer_twos import EnvType
import gymnasium as gym
from ray import tune
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


tune.registry.register_env("Soccer", create_rllib_env)

config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
config = config.resources(num_gpus=1)
config = config.rollouts(num_rollout_workers=1)
config = config.framework("torch")
config.env_config = {"render": False, "time_scale": 50, "multiagent": False, "variation": EnvType.team_vs_policy,
                     "flatten_branched": True, "single_player": True}


algo = config.build(env="Soccer")  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.