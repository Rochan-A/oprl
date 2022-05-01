from __future__ import division
import numpy as np
from utils import *

#############################
#
# Source: https://www.johndcook.com/blog/standard_deviation/,
# https://stackoverflow.com/a/45949178
#
#############################

import collections
import math

class RunningStats:
    def __init__(self, WIN_SIZE=20):
        self.n = 0
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.windows = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):

        self.windows.append(x)

        if self.n <= self.WIN_SIZE:
            # Calculating first variance
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.run_var += delta * (x - self.mean)
        else:
            # Adjusting variance
            x_removed = self.windows.popleft()
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)

    def get_mean(self):
        return self.mean if self.n else 0.0

    def get_var(self):
        return self.run_var / (self.WIN_SIZE - 1) if self.n > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.windows)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))


class ActionValuePolicy():
    '''
    Deterministic policy based on action values:
        pi(s,a) ~ Q(s,a)
    '''

    def __init__(self, nStates, nActions, actionSelectionMethod="greedy", epsilon=0.0, tieBreakingMethod="consistent"):
        self.nStates = nStates
        self.nActions = nActions
        self.actionSelectionMethod = actionSelectionMethod
        self.epsilon = epsilon
        self.tieBreakingMethod = tieBreakingMethod
        if(self.tieBreakingMethod=="arbitrary"):
            self.argmax_function = argmax
        elif(self.tieBreakingMethod=="consistent"):
            self.argmax_function = np.argmax
        else:
            sys.exit("ERROR: ActionValuePolicy: tieBreakingMethod not recognized!")
        if(self.actionSelectionMethod=="egreedy"):
            self.actionSelection_function = selectAction_egreedy
        elif(self.actionSelectionMethod=="softmax"):
            self.actionSelection_function = selectAction_softmax
        elif(self.actionSelectionMethod=="greedy"):
            self.actionSelection_function = selectAction_greedy
        elif(self.actionSelectionMethod=="esoft"):
            self.actionSelection_function = selectAction_esoft
        else:
            sys.exit("ERROR: ActionValuePolicy: actionSelectionMethod not recognized!")
        self.actionValueTable = np.zeros([nStates, nActions], dtype=float)
        self.normalization_function = normalize_sum

    def selectAction(self, state, actionsAvailable=None):
        if(actionsAvailable is None):
            actionValues = self.actionValueTable[state,:]
            actionList = np.array(range(self.nActions))
        else:
            actionValues = self.actionValueTable[state, actionsAvailable]
            actionList = np.array(actionsAvailable)
        actionIdx = self.actionSelection_function(actionValues, argmaxfun=self.argmax_function, epsilon=self.epsilon)
        return actionList[actionIdx]

    def update(self, state, actionValues):
        self.actionValueTable[state,:] = actionValues

    def getProbability(self, state, action=None):
        p = self.normalization_function(self.actionValueTable[state,:], argmaxfun=self.argmax_function)
        if(action is None):
            return p
        else:
            return p[action]

    def reset(self):
        self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)

class TDControlAgent:

    def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        self.name = "Generic TDControlAgent"
        self.nStates = nStates
        self.nActions = nActions
        self.alpha = alpha
        self.gamma = gamma
        self.valueInit = valueInit
        if(self.valueInit=="zeros"):
            self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable = np.random.rand(self.nStates, self.nActions)
        else:
            sys.exit("ERROR: TDControlAgent: valueInit not recognized!")
        self.policy = ActionValuePolicy(self.nStates, self.nActions, actionSelectionMethod=actionSelectionMethod,
            epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
        for idx_state in range(self.nStates):
            self.policy.update(idx_state, self.actionValueTable[idx_state,:])

    def selectAction(self, state, actionsAvailable=None):
        return self.policy.selectAction(state, actionsAvailable)

    def getGreedyAction(self, state, actionsAvailable=None):
        if(actionsAvailable is None):
            actionValues = self.actionValueTable[state,:]
            actionList = np.array(range(self.nActions))
        else:
            actionValues = self.actionValueTable[state, actionsAvailable]
            actionList = np.array(actionsAvailable)
        actionIdx = selectAction_greedy(actionValues)
        return actionList[actionIdx]

    def getValue(self, state):
        return np.dot(self.policy.getProbability(state), self.actionValueTable[state,:])

    def getActionValue(self, state, action):
        return self.actionValueTable[state,action]

    def getName(self):
        return self.name

    def reset(self):
        if(self.valueInit=="zeros"):
            self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable = np.random.rand(self.nStates, self.nActions)
        else:
            sys.exit("ERROR: TDControlAgent: valueInit not recognized!")
        self.policy.reset()
        for idx_state in range(self.nStates):
            self.policy.update(idx_state, self.actionValueTable[idx_state,:])

class QLearning(TDControlAgent):

    def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon=epsilon,
            tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
        self.name = "Q-Learning"

    def update(self, episode):
        maxTDError = 0.0
        T = len(episode)
        for t in range(0, T-1):
            state = episode[t]["state"]
            action = episode[t]["action"]
            reward = episode[t+1]["reward"]
            next_state = episode[t+1]["state"]
            if("allowedActions" in episode[t+1].keys()):
                allowedActions = episode[t+1]["allowedActions"]
            else:
                allowedActions = np.array(range(self.nActions))
            td_error = reward + self.gamma * np.max(self.actionValueTable[next_state,allowedActions]) - self.actionValueTable[state, action]
            self.actionValueTable[state, action] += self.alpha * td_error
            self.policy.update(state, self.actionValueTable[state,:])
            maxTDError = max(maxTDError, abs(td_error))
        return maxTDError

class DoubleQLearning(TDControlAgent):

    def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        self.name = "Double Q-Learning"
        self.nStates = nStates
        self.nActions = nActions
        self.alpha = alpha
        self.gamma = gamma
        self.tieBreakingMethod = tieBreakingMethod
        self.valueInit = valueInit
        if(self.tieBreakingMethod=="arbitrary"):
            self.argmax_function = argmax
        elif(self.tieBreakingMethod=="consistent"):
            self.argmax_function = np.argmax
        else:
            sys.exit("ERROR: DoubleQLearning: tieBreakingMethod not recognized!")
        if(self.valueInit=="zeros"):
            self.actionValueTable_1 = np.zeros([self.nStates, self.nActions], dtype=float)
            self.actionValueTable_2 = np.zeros([self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable_1 = np.random.rand(self.nStates, self.nActions)
            self.actionValueTable_2 = np.random.rand(self.nStates, self.nActions)
        else:
            sys.exit("ERROR: DoubleQLearning: valueInit not recognized!")
        self.policy = ActionValuePolicy(self.nStates, self.nActions,
            actionSelectionMethod=actionSelectionMethod, epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
        for idx_state in range(self.nStates):
            self.policy.update(idx_state, self.actionValueTable_1[idx_state,:]+self.actionValueTable_2[idx_state,:])

    def update(self, episode):
        T = len(episode)
        for t in range(0, T-1):
            state = episode[t]["state"]
            action = episode[t]["action"]
            reward = episode[t+1]["reward"]
            next_state = episode[t+1]["state"]
            if("allowedActions" in episode[t+1].keys()):
                allowedActions = episode[t+1]["allowedActions"]
            else:
                allowedActions = np.array(range(self.nActions))
            if(np.random.rand()<0.5):
                next_action = self.argmax_function(self.actionValueTable_1[next_state, allowedActions])
                td_error = reward + self.gamma * self.actionValueTable_2[next_state, next_action] - self.actionValueTable_1[state, action]
                self.actionValueTable_1[state, action] += self.alpha * td_error
            else:
                next_action = self.argmax_function(self.actionValueTable_2[next_state,allowedActions])
                td_error = reward + self.gamma * self.actionValueTable_1[next_state,next_action] - self.actionValueTable_2[state, action]
                self.actionValueTable_2[state, action] += self.alpha * td_error
            self.policy.update(state, (self.actionValueTable_1[state,:]+self.actionValueTable_2[state,:]))

    def getValue(self, state):
        q_values = self.actionValueTable_1[state,:] + self.actionValueTable_2[state,:]
        return np.dot(self.policy.getProbability(state), q_values)

    def getActionValue(self, state, action):
        return self.actionValueTable_1[state,action] + self.actionValueTable_2[state,action]

    def getGreedyAction(self, state, actionsAvailable=None):
        actionValueTable = (self.actionValueTable_1 + self.actionValueTable_2)/2.0
        if(actionsAvailable is None):
            actionValues = actionValueTable[state,:]
            actionList = np.array(range(self.nActions))
        else:
            actionValues = actionValueTable[state, actionsAvailable]
            actionList = np.array(actionsAvailable)
        actionIdx = selectAction_greedy(actionValues)
        return actionList[actionIdx]

    def reset(self):
        if(self.valueInit=="zeros"):
            self.actionValueTable_1 = np.zeros([self.nStates, self.nActions], dtype=float)
            self.actionValueTable_2 = np.zeros([self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable_1 = np.random.rand(self.nStates, self.nActions)
            self.actionValueTable_2 = np.random.rand(self.nStates, self.nActions)
        else:
            sys.exit("ERROR: DoubleQLearning: valueInit not recognized!")
        self.policy.reset()
        for idx_state in range(self.nStates):
            self.policy.update(idx_state, self.actionValueTable_1[idx_state,:]+self.actionValueTable_2[idx_state,:])


class PessimisticQLearning(TDControlAgent):

    def __init__(self, nStates, nActions, alpha, gamma, pessimism_coeff=0.1, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon=epsilon,
            tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
        self.name = "Pessimistic Q-Learning"
        self.pessimism_coeff = pessimism_coeff
        self.visit = np.ones((nStates, nActions))

    def update(self, episode):
        maxTDError = 0.0
        T = len(episode)
        for t in range(0, T-1):
            state = episode[t]["state"]
            action = episode[t]["action"]
            reward = episode[t+1]["reward"]
            next_state = episode[t+1]["state"]
            if("allowedActions" in episode[t+1].keys()):
                allowedActions = episode[t+1]["allowedActions"]
            else:
                allowedActions = np.array(range(self.nActions))

            self.visit[state, action] += 1

            error = (
                    reward
                    + self.gamma
                    * np.max(self.actionValueTable[next_state,allowedActions])
                    - self.actionValueTable[state, action]
                    - (self.pessimism_coeff/self.visit[state, action])
            )

            self.actionValueTable[state, action] += self.alpha * error
            self.policy.update(state, self.actionValueTable[state,:])
            maxTDError = max(maxTDError, abs(error))
        return maxTDError


class MaxminQLearning(TDControlAgent):

    def __init__(self, nStates, nActions, alpha, gamma, n=2, buffer_size=100, mini_batch=50, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        self.name = "Maxmin Q-Learning"
        self.nStates = nStates
        self.nActions = nActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionSelectionMethod = actionSelectionMethod

        self.n = n # Number of Q functions
        self.buffer_size = buffer_size
        self.mini_batch = mini_batch

        self.tieBreakingMethod = tieBreakingMethod
        self.valueInit = valueInit
        if(self.tieBreakingMethod=="arbitrary"):
            self.argmax_function = argmax
        elif(self.tieBreakingMethod=="consistent"):
            self.argmax_function = np.argmax
        else:
            sys.exit("ERROR: MaxminQLearning: tieBreakingMethod not recognized!")
        if(self.valueInit=="zeros"):
            self.actionValueTable = np.zeros([self.n, self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable = np.random.rand(self.n, self.nStates, self.nActions)
        else:
            sys.exit("ERROR: MaxminQLearning: valueInit not recognized!")
        self.policy = ActionValuePolicy(self.nStates, self.nActions,
            actionSelectionMethod=self.actionSelectionMethod, epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)

        for idx_state in range(self.nStates):
            self.policy.update(idx_state,
                self.actionValueTable[
                    np.argmin(np.sum(self.actionValueTable[:, idx_state,: ])), idx_state,:]
            )

    def update(self, episode):
        T = len(episode)
        for batch in range(0, T-1, self.buffer_size):

            # select random q function to update
            q_choice = np.random.randint(0, self.n)

            # select random mini-batch. NOTE: Paper does not specify the size
            # of the mini-batch, using 0.5*len(buffer)
            mini_batch_idx = np.random.randint(0, min(self.mini_batch, len(episode[batch:]))-1, (self.mini_batch))

            for t_d in mini_batch_idx:
                t = batch + t_d

                state = episode[t]["state"]
                action = episode[t]["action"]
                reward = episode[t+1]["reward"]
                next_state = episode[t+1]["state"]
                if("allowedActions" in episode[t+1].keys()):
                    allowedActions = episode[t+1]["allowedActions"]
                else:
                    allowedActions = np.array(range(self.nActions))

                error = (
                        reward
                        + self.gamma
                        * np.max(self.actionValueTable[q_choice, next_state, allowedActions])
                        - self.actionValueTable[q_choice, state, action]
                )

                self.actionValueTable[q_choice, state, action] += self.alpha * error

                # Update policy with min q function
                self.policy.update(state, self.actionValueTable[np.argmin(np.sum(self.actionValueTable[:, state,: ])), state,:])


    def getValue(self, state):
        q_values = self.actionValueTable[np.argmin(np.sum(self.actionValueTable[:, state,: ])), state,:]
        return np.dot(self.policy.getProbability(state), q_values)

    def getActionValue(self, state, action):
        return self.actionValueTable[np.argmin(np.sum(self.actionValueTable[:, state,: ])), state, action]

    def getGreedyAction(self, state, actionsAvailable=None):
        actionValueTable = self.actionValueTable[np.argmin(np.sum(self.actionValueTable[:, state,: ])), ::]
        if(actionsAvailable is None):
            actionValues = actionValueTable[state,:]
            actionList = np.array(range(self.nActions))
        else:
            actionValues = actionValueTable[state, actionsAvailable]
            actionList = np.array(actionsAvailable)
        actionIdx = selectAction_greedy(actionValues)
        return actionList[actionIdx]

    def reset(self):
        if(self.valueInit=="zeros"):
            self.actionValueTable = np.zeros([self.n, self.nStates, self.nActions], dtype=float)
        elif(self.valueInit=="random"):
            self.actionValueTable = np.random.rand(self.n, self.nStates, self.nActions)
        else:
            sys.exit("ERROR: MaxminQLearning: valueInit not recognized!")
        self.policy = ActionValuePolicy(self.nStates, self.nActions,
            actionSelectionMethod=self.actionSelectionMethod, epsilon=self.epsilon, tieBreakingMethod=self.tieBreakingMethod)


class MeanVarQLearning(TDControlAgent):

    def __init__(self, nStates, nActions, alpha, gamma, coeff=0.1, actionSelectionMethod="egreedy", epsilon=0.01,
        tieBreakingMethod="arbitrary", valueInit="zeros"):
        super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon=epsilon,
            tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
        self.name = "MeanVar Q-Learning"
        self.coeff = coeff
        self.rs = [[RunningStats(20) for i in range(nActions)] for k in range(nStates)] # window size

    def update(self, episode):
        maxTDError = 0.0
        T = len(episode)
        for t in range(0, T-1):
            state = episode[t]["state"]
            action = episode[t]["action"]
            reward = episode[t+1]["reward"]
            next_state = episode[t+1]["state"]
            if("allowedActions" in episode[t+1].keys()):
                allowedActions = episode[t+1]["allowedActions"]
            else:
                allowedActions = np.array(range(self.nActions))

            self.rs[state][action].push(reward)
            error = (
                    reward
                    + self.gamma
                    * np.max(self.actionValueTable[next_state,allowedActions])
                    - self.actionValueTable[state, action]
                    - (0.1 * self.rs[state][action].n * self.rs[state][action].get_var())
            )

            self.actionValueTable[state, action] += self.alpha * error
            self.policy.update(state, self.actionValueTable[state,:])
            maxTDError = max(maxTDError, abs(error))
        return maxTDError
