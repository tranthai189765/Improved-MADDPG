import mate

env = mate.make('MultiAgentTracking-v0')
env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
env.seed(0)
done = False
target_joint_observation = env.reset()
while not done:
    # env.render()
    target_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    target_joint_observation, target_team_reward, done, target_infos = env.step(target_joint_action)
    print("done = ", done)