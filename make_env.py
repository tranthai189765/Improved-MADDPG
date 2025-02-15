import mate 

def make_env():
    env = mate.make('MultiAgentTracking-v0')
    env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
    env.seed(0)

    return env