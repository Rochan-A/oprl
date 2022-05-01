#########################################################################################
# Credits: Most of the code in this directory has been borrowed from                    #
# https://github.com/cemkaraoguz/reinforcement-learning-an-introduction-second-edition  #
# Modifications were made to the original authors code for our use.                     #
#########################################################################################

import sys
from matplotlib.pyplot import tight_layout
import numpy as np
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from utils import *
from env import *
from algos import *
import seaborn as sns

def runExperiment(nEpisodes, env, agent):
    countLeftFromA = np.zeros(nEpisodes)
    for e in range(nEpisodes):

        state = env.reset()
        done = False
        while not done:

            experiences = [{}]

            action = agent.selectAction(state, env.getAvailableActions())

            experiences[-1]['state'] = state
            experiences[-1]['action'] = action
            experiences[-1]['done'] = done

            if((state==env.STATE_A) and (action==env.ACTION_LEFT)):
                countLeftFromA[e] += 1

            new_state, reward, done = env.step(action)

            xp = {}
            xp['reward'] = reward
            xp['state'] = new_state
            xp['done'] = done
            xp['allowedActions'] = env.getAvailableActions(new_state)
            experiences.append(xp)

            agent.update(experiences)

            state = new_state

    return countLeftFromA


if __name__=="__main__":

    nExperiments = 1000
    nEpisodes = 300

    # Agents
    alpha_QLearning = 0.1
    gamma_QLearning = 1.0
    alpha_DoubleQLearning = 0.1
    gamma_DoubleQLearning = 1.0

    # Policy
    epsilon_QLearning = 0.1
    epsilon_DoubleQLearning = 0.1

    # Environment
    env = MaximizationBias(mu=-0.1)#MaximizationBias(mu=-0.1)
    q_counts = [2, 3, 4]

    #env.printEnv()

    allCountLeftFromA_QLearning = np.zeros(nEpisodes)
    allCountLeftFromA_DoubleQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_PessimisticQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_MaxminQLearning = [np.zeros(nEpisodes) for _ in q_counts]
    allCountLeftFromA_MeanVarQLearning = np.zeros(nEpisodes)
    for idx_experiment in tqdm(range(nExperiments)):

        agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
        agent_PessimisticQLearning = PessimisticQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_MeanVarQLearning = MeanVarQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)

        countLeftFromA_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
        countLeftFromA_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
        countLeftFromA_PessimisticQLearning = runExperiment(nEpisodes, env, agent_PessimisticQLearning)
        countLeftFromA_MeanVarQLearning = runExperiment(nEpisodes, env, agent_MeanVarQLearning)

        allCountLeftFromA_QLearning += countLeftFromA_QLearning
        allCountLeftFromA_DoubleQLearning += countLeftFromA_DoubleQLearning
        allCountLeftFromA_PessimisticQLearning += countLeftFromA_PessimisticQLearning
        allCountLeftFromA_MeanVarQLearning += countLeftFromA_MeanVarQLearning

        for idx, q_count in enumerate(q_counts):
            agent_MaxminQLearning = MaxminQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, n=q_count, epsilon=epsilon_QLearning)
            countLeftFromA_MaxminQLearning = runExperiment(nEpisodes, env, agent_MaxminQLearning)
            allCountLeftFromA_MaxminQLearning[idx] += countLeftFromA_MaxminQLearning

    fig = pl.figure(figsize=(10, 10), dpi=200)

    clrs = sns.color_palette("husl", 5 + len(q_counts))
    with sns.axes_style("darkgrid"):
        pl.plot(allCountLeftFromA_QLearning/nExperiments*100, label="Q-Learning", c=clrs[0])
        pl.plot(allCountLeftFromA_DoubleQLearning/nExperiments*100, label="Double Q-Learning", c=clrs[1])
        pl.plot(allCountLeftFromA_PessimisticQLearning/nExperiments*100, label="Pessimistic Q-Learning", c=clrs[2])
        pl.plot(allCountLeftFromA_MeanVarQLearning/nExperiments*100, label="MeanVar Q-Learning", c=clrs[3])

        for idx, val in enumerate(q_counts):
            pl.plot(allCountLeftFromA_MaxminQLearning[idx]/nExperiments*100, label="Maxmin Q-Learning (N={})".format(val), c=clrs[4+idx])

        pl.plot(np.ones(nEpisodes)*5.0, c=clrs[-1])

    pl.xlabel("Episodes")
    pl.ylabel("% left actions from A")
    pl.legend()
    pl.tight_layout()
    pl.savefig('plot_visits.png')