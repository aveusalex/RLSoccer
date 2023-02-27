from ray.rllib.algorithms.ppo import PPOConfig
from soccer_twos import EnvType
import gymnasium as gym
from ray import tune
from ray.rllib import MultiAgentEnv
import soccer_twos
from ray.tune.logger import pretty_print


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


def train():
    tune.registry.register_env("Soccer", create_rllib_env)

    config = PPOConfig()
    config = config.training(gamma=0.9, lr=1, kl_coeff=0.3, sgd_minibatch_size=1000)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=4, create_env_on_local_worker=True)
    config = config.framework("torch")
    config.env_config = {"render": False, "time_scale": 50, "multiagent": False, "variation": EnvType.team_vs_policy,
                         "flatten_branched": True, "single_player": True}

    evaluation_config = {"render": True, "time_scale": 1, "variation": EnvType.team_vs_policy,
                                "flatten_branched": True, "single_player": True}

    config.evaluation(evaluation_config=evaluation_config)

    algo = config.build(env="Soccer")  # 2. build the algorithm,

    for i in range(1):
        result = algo.train()  # 3. train it,
        print(f"iter {i}:\n", pretty_print(result), "\n\n")

    return algo


def test(algo):
    # testando o agente
    env = soccer_twos.make(render=True, time_scale=1, variation=EnvType.team_vs_policy,
                           flatten_branched=True, single_player=True)

    reward = 0
    obs = env.reset()
    while True:
        action = algo.compute_action(obs)
        obs, reward, done, info = env.step(action)

        # input("Press Enter to continue...")
        reward += reward

        if done["__all__"]:
            print("Total Reward: ", reward)
            env.reset()


if __name__ == "__main__":
    agent = train()
    test(agent)