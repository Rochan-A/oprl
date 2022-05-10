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
import random

def runExperiment(nEpisodes, env, agent):
    countLeftFromA = np.zeros(nEpisodes)
    rewards = np.zeros(nEpisodes)
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

            rewards[e] += e
            state = new_state

    return countLeftFromA #, rewards


def runExperiment_Bandits(nEpisodes, env, agent):
    countLeftFromA = np.zeros(nEpisodes)
    estimators = np.zeros(nEpisodes)
    #rs = [[RunningStats(20) for i in range(agent.nActions)] for k in range(agent.nStates)]
    for e in range(nEpisodes):

        state = env.reset()
        done = False
        i = 0
        c_r = 0
        while not done:

            experiences = [{}]

            action = agent.selectAction(state, env.getAvailableActions())

            experiences[-1]['state'] = state
            experiences[-1]['action'] = action
            experiences[-1]['done'] = done

            if((state==env.STATE_A) and (action==env.ACTION_LEFT)):
                countLeftFromA[e] += 1

            new_state, reward, done = env.step(action)

            agent.memory.append([state, action, reward, new_state, done])

            for j in range(agent.buffer_size):
                update_ind = np.random.choice( agent.max_estimators )
                update_trans = random.sample(agent.memory, 1)[0]
                a_p = np.argmax(agent.policy.actionValueTable[update_trans[3], env.getAvailableActions(update_trans[3])])

                error = (
                    update_trans[2]
                    + agent.gamma
                    * agent.policy.actionValueTable[update_trans[3], a_p]
                    - agent.actionValueTable[update_ind, update_trans[0], update_trans[1]]
                )

                agent.actionValueTable[update_ind, update_trans[0], update_trans[1]] += agent.alpha * error

            c_r += reward

            state = new_state

            agent.get_min_q(agent.active_estimators)

        estimators[e] = agent.active_estimators
        agent.c_r_memory.append([c_r, agent.active_estimators])
        if len(agent.c_r_memory) > 5:
            new_reward = (np.sum( list(agent.c_r_memory), axis=0 )[0] - agent.c_r_memory[0][0]) * 1.0 / 5
            num_active_estimators = agent.c_r_memory[0][1] - 1
            agent.q_est[num_active_estimators] = agent.q_est[num_active_estimators] + 0.05 * (new_reward - agent.q_est[num_active_estimators])
            agent.num_a_est[ num_active_estimators ] += 1

        agent.active_estimators = agent.get_active_estimators(e+2)

    return countLeftFromA, estimators


def runExperiment_Bandits2(nEpisodes, env, agent):
    countLeftFromA = np.zeros(nEpisodes)


    rs = [[RunningStats(20) for i in range(agent.nActions)] for k in range(agent.nStates)]
    for e in range(nEpisodes):

        state = env.reset()
        done = False
        i = 0
        c_r = 0
        while not done:

            experiences = [{}]

            action = agent.selectAction(state, env.getAvailableActions())

            experiences[-1]['state'] = state
            experiences[-1]['action'] = action
            experiences[-1]['done'] = done

            if((state==env.STATE_A) and (action==env.ACTION_LEFT)):
                countLeftFromA[e] += 1

            new_state, reward, done = env.step(action)
            rs[state][action].push(reward)

            agent.memory.append([state, action, reward, new_state, done])

            for j in range(agent.buffer_size):
                update_ind = np.random.choice( agent.max_estimators )
                update_trans = random.sample(agent.memory, 1)[0]
                a_p = np.argmax(agent.policy.actionValueTable[update_trans[3], env.getAvailableActions(update_trans[3])])

                error = (
                    update_trans[2]
                    + agent.gamma
                    * agent.policy.actionValueTable[update_trans[3], a_p]
                    - agent.actionValueTable[update_ind, update_trans[0], update_trans[1]]
                )

                agent.actionValueTable[update_ind, update_trans[0], update_trans[1]] += agent.alpha * error

            c_r += reward

            state = new_state

            agent.get_min_q(agent.active_estimators)

        agent.c_r_memory.append([c_r, agent.active_estimators])
        if len(agent.c_r_memory) > 5:
            new_reward = (np.sum( list(agent.c_r_memory), axis=0 )[0] - agent.c_r_memory[0][0]) * 1.0 / 5
            num_active_estimators = agent.c_r_memory[0][1]-1
            agent.q_est[num_active_estimators] = agent.q_est[num_active_estimators] + 0.05 * (new_reward - agent.q_est[num_active_estimators])
            agent.num_a_est[ num_active_estimators ] += 1

    return countLeftFromA


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_exp(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


if __name__=="__main__":

    nExperiments = 100
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
    env = MaximizationBias(mu=-0.1)
    q_counts = [2, 4, 6]


    allCountLeftFromA_QLearning = np.zeros(nEpisodes)
    allCountLeftFromA_DoubleQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_PessimisticQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_MaxminQLearning = [np.zeros(nEpisodes) for _ in q_counts]
    allCountLeftFromA_MeanVarQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_MaxminBanditQLearning = np.zeros(nEpisodes)
    allCountLeftFromA_MaxminBanditQLearning2 = np.zeros(nEpisodes)

    avg_estimator = np.zeros(nEpisodes)

    for idx_experiment in tqdm(range(nExperiments)):

        agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
        agent_PessimisticQLearning = PessimisticQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_MeanVarQLearning = MeanVarQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_MaxminBanditQLearning = MaxminBanditQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_MaxminBanditQLearning2 = MaxminBanditQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)

        countLeftFromA_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
        countLeftFromA_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
        countLeftFromA_PessimisticQLearning = runExperiment(nEpisodes, env, agent_PessimisticQLearning)
        countLeftFromA_MeanVarQLearning = runExperiment(nEpisodes, env, agent_MeanVarQLearning)

        countLeftFromA_MaxminBanditQLearning, estimators = runExperiment_Bandits(nEpisodes, env, agent_MaxminBanditQLearning)
        countLeftFromA_MaxminBanditQLearning2 = np.zeros(nEpisodes) #runExperiment_Bandits2(nEpisodes, env, agent_MaxminBanditQLearning2)

        allCountLeftFromA_QLearning += countLeftFromA_QLearning
        allCountLeftFromA_DoubleQLearning += countLeftFromA_DoubleQLearning
        allCountLeftFromA_PessimisticQLearning += countLeftFromA_PessimisticQLearning
        allCountLeftFromA_MeanVarQLearning += countLeftFromA_MeanVarQLearning
        allCountLeftFromA_MaxminBanditQLearning += countLeftFromA_MaxminBanditQLearning
        allCountLeftFromA_MaxminBanditQLearning2 += countLeftFromA_MaxminBanditQLearning2

        avg_estimator += estimators

        for idx, q_count in enumerate(q_counts):
            agent_MaxminQLearning = MaxminQLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, n=q_count, epsilon=epsilon_QLearning)
            countLeftFromA_MaxminQLearning = runExperiment(nEpisodes, env, agent_MaxminQLearning)
            allCountLeftFromA_MaxminQLearning[idx] += countLeftFromA_MaxminQLearning

    fig = pl.figure(figsize=(10, 10), dpi=200)

    clrs = sns.color_palette("husl", 7 + len(q_counts))
    with sns.axes_style("darkgrid"):
        pl.plot(allCountLeftFromA_QLearning/nExperiments*100, label="Q-Learning", c=clrs[0])
        pl.plot(allCountLeftFromA_DoubleQLearning/nExperiments*100, label="Double Q-Learning", c=clrs[1])
        pl.plot(allCountLeftFromA_PessimisticQLearning/nExperiments*100, label="Pessimistic Q-Learning", c=clrs[2])
        # pl.plot(allCountLeftFromA_MeanVarQLearning/nExperiments*100, label="MeanVar Q-Learning", c=clrs[3])
        pl.plot(allCountLeftFromA_MaxminBanditQLearning/nExperiments*100, label="Maxmin Bandit Q-Learning", c=clrs[4])

        for idx, val in enumerate(q_counts):
            pl.plot(allCountLeftFromA_MaxminQLearning[idx]/nExperiments*100, label="Maxmin Q-Learning (N={})".format(val), c=clrs[6+idx])

        pl.plot(np.ones(nEpisodes)*5.0, c=clrs[-1])

        pl.xlabel("Episodes")
        pl.ylabel("% left actions from A")
        pl.legend()
        pl.tight_layout()
        pl.savefig('plot_visits.png')

    fig = pl.figure(figsize=(10, 10), dpi=200)

    clrs = sns.color_palette("husl", 7 + len(q_counts))
    with sns.axes_style("darkgrid"):
        pl.plot(smooth(allCountLeftFromA_MaxminBanditQLearning/nExperiments*100, 10), label="Maxmin Bandit Q-Learning", c=clrs[4])
        pl.plot(smooth_exp(allCountLeftFromA_MaxminBanditQLearning2/nExperiments*100, 0.9), label="Maxmin Bandit Q-Learning (2)", c=clrs[5])

        pl.plot(np.ones(nEpisodes)*5.0, c=clrs[-1])

        pl.xlabel("Episodes")
        pl.ylabel("% left actions from A")
        pl.legend()
        pl.tight_layout()
        pl.savefig('plot_visits_maxmin_bandit.png')