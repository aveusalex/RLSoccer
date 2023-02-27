import soccer_twos

env = soccer_twos.make(render=True, time_scale=1)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0
while True:
    act1 = env.action_space.sample()
    act2 = env.action_space.sample()
    act3 = env.action_space.sample()
    act4 = env.action_space.sample()

    obs, reward, done, info = env.step(
        {
            0: act1,
            1: act2,
            2: act3,
            3: act4,
        }
    )

    print("obs:", obs[0].shape)
    print("reward:", reward)
    print("done:", done)
    print("info:", info[0].keys())

    # input("Press Enter to continue...")
    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
