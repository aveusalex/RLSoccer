import soccer_twos
from soccer_twos import EnvType


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


