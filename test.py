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
    status = env.reset()
    for t in range(2000):
        action = agent.step(status)
        status, reward = env.step(action)
        plotter.update() 
