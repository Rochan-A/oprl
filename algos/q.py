from typing import Tuple
import numpy as np
import gym
import random

from collections import deque

from policies.q_values import EGPolicy
from easydict import EasyDict



def Q_learning(
    env: gym.Env,
    n: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    decay: float,
    interval: int,
    Q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epslion: epsilon-greedy exploration
        decay: epsilon decay
        interval: epsilon update interval
        Q: initial Q function
        rng:
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (r, #s) logs
    """

    #####################
    # Q Learning (Hint: Sutton Book p. 131)
    #####################

    # Log Q value, cummulative reward, # of steps
    Qlogger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n,2))

    terminal = env.final_state
    Q[terminal, :] = 0

    pi = EGPolicy(Q, epsilon)

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + (gamma * np.max(Q[s1, :])) - Q[s, a])
            s = s1

            pi.update(Q, epsilon)

            c_r += r
        if i % interval == 0:
            epsilon *= decay
        Qlogger[i, ::] = Q
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return Q, Qlogger, np.int64(Envlogs)


def DoubleQ(
    env: gym.Env,
    n: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    decay: float,
    interval: int,
    Q1: np.ndarray,
    Q2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epsilon: epsilon-greedy exploration
        decay: epsilon decay
        interval: epsilon decay interval
        Q1: initial Q1 function
        Q2: initial Q2 function
    ret:
        Q1: Q1 function; numpy array shape of [nS,nA]
        Q1: Q2 function; numpy array shape of [nS,nA]
        Qlogs1: Q1 logs
        Qlogs2: Q2 logs
        envlogs: (r, #s) logs
    """

    #####################
    # Double Q Learning (Hint: Sutton Book p. 135-136)
    #####################
    # Log Q value, cummulative reward, # of steps
    Qlogger1 = np.empty((n, env.nS, env.nA,))
    Qlogger2 = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n,2))

    terminal = env.final_state
    Q1[terminal, :] = 0
    Q2[terminal, :] = 0

    # Keep track of which policy we are using
    # (see https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
    if np.random.random() < 0.5:
        pi = EGPolicy(Q1, epsilon)
        A = True
    else:
        pi = EGPolicy(Q2, epsilon)
        A = False

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)

            if A:
                Q1[s, a] += alpha * (
                    r + gamma * Q2[s1, np.argmax(Q1[s1, :])] - Q1[s, a]
                )
            else:
                Q2[s, a] += alpha * (
                    r + gamma * Q1[s1, np.argmax(Q2[s1, :])] - Q2[s, a]
                )

            s = s1

            if np.random.random() < 0.5:
                pi.update(Q1, epsilon)
                A = True
            else:
                pi.update(Q2, epsilon)
                A = False

            c_r += r
        if i % interval == 0:
            epsilon *= decay
        Qlogger1[i, ::] = Q1
        Qlogger2[i, ::] = Q2
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return Q1, Q2, Qlogger1, Qlogger2, np.int64(Envlogs)


def get_min_q(MMQ: np.array):
    "Returns the minimum Q table"
    return np.amin(MMQ[:, :, :], axis=0).squeeze()



def MaxminQ(env: gym.Env, config, Q_init: np.array,):
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epsilon: epsilon-greedy exploration
        decay: epsilon decay
        interval: epsilon decay interval
        estimators: Number of Q tables
        Q_init: Q value initializer
    ret:
        Q1: Q1 function; numpy array shape of [nS,nA]
        Q1: Q2 function; numpy array shape of [nS,nA]
        Qlogs1: Q1 logs
        Qlogs2: Q2 logs
        envlogs: (r, #s) logs
    """

    MMQ_logger = np.empty(( config.mmq_learning.steps , config.mmq_learning.estimators, env.nS, env.nA,))
    Envlogs = np.empty(( config.mmq_learning.steps , 2))

    MMQ = np.repeat( Q_init[np.newaxis, :, :], config.mmq_learning.estimators,  axis=0)
    assert MMQ.shape == (config.mmq_learning.estimators, env.nS, env.nA)

    terminal = env.final_state
    MMQ[:, terminal, :] = 0

    Q_min = get_min_q(MMQ)
    assert Q_min.shape == (env.nS, env.nA)

    epsilon = config.mmq_learning.epsilon
    pi = EGPolicy(Q_min, epsilon)

    memory = deque([], maxlen=config.mmq_learning.buffer_size)

    for i in range( config.mmq_learning.steps ):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)
            memory.append([s, a, r, s1])

            for j in range(config.mmq_learning.replay_size):
                update_ind = np.random.choice(config.mmq_learning.estimators)
                update_trans = random.sample(memory, 1)[0]
                MMQ[update_ind, update_trans[0], update_trans[1]] +=  config.mmq_learning.alpha * (update_trans[2] + config.gamma * Q_min[update_trans[3], np.argmax(Q_min[update_trans[3], :])] - MMQ[update_ind, update_trans[0], update_trans[1]] )

            s = s1

            Q_min = get_min_q(MMQ)
            pi.update(Q_min, epsilon)

            c_r += r
        if i % config.mmq_learning.interval == 0:
            epsilon *= config.mmq_learning.decay

        MMQ_logger[i, :, :, :] = MMQ
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    print(MMQ_logger.shape)

    return MMQ, MMQ_logger, np.int64(Envlogs)




def PessimisticQ(
    env: gym.Env,
    n: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    decay: float,
    interval: int,
    pessimism_coeff: float,
    PQ: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epslion: epsilon-greedy exploration
        decay: epsilon decay
        interval: epsilon update interval
        Q: initial Q function
        rng:
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (r, #s) logs
    """

    #####################
    # Pessimistic Q learning (ref. https://arxiv.org/pdf/2202.13890.pdf)
    #####################

    # Log Q value, cummulative reward, # of steps
    Qlogger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n,2))

    pi = EGPolicy(PQ, epsilon)

    visit = np.zeros(PQ.shape) + 1
    vl = []

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)

            visit[s, a] += 1
            PQ[s, a] += alpha * (
                r
                + gamma * np.max(PQ[s1, :])
                - PQ[s, a]
                - (pessimism_coeff / visit[s, a])
            )
            s = s1
            pi.update(PQ, epsilon)

            c_r += r
        if i % interval == 0:
            epsilon *= decay

        Qlogger[i, ::] = PQ
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count
    return PQ, Qlogger, np.int64(Envlogs)
