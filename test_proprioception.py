import numpy as np

from bbo import BBO
from arena_env import Agent, ArenaEnv, Plotter
import matplotlib.pyplot as plt


if __name__ == "__main__":
    

    plt.ion()
    
    num_rollouts = 30
    lmb = 0.0001
    stime = 150
    sigma = 0.001
    epochs = 200
    num_agent_units = 200

    env = ArenaEnv()
    agent = Agent.load("agent.dump")    


    plotter = Plotter(env)
    env.reset()
    status = np.zeros(2)
    for t in range(2000):
        a = t/2000
        b = 1 - a
        action = agent.step(np.hstack([a,b,status]))
        _, reward = env.step(action)
        status = np.copy(action)
        plotter.update() 
