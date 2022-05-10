from distutils.command.upload import upload
from typing import Tuple
import numpy as np
import gym
import copy

from collections import deque
from policies.tilecoding import TileCoder

from policies.q_values import QLearningAgent, SA, S, update #, QLearningAgent_Maxmin
from easydict import EasyDict

from utils.buffers import RunningStats
from numpy_ringbuffer import RingBuffer


def Q_learning(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    T: TileCoder
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: configs
        Q: initial Q function
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: reward logs
        visits: state visit logs
    """

    #####################
    # Q Learning (Hint: Sutton Book p. 131)
    #####################

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    alpha = config.q_learning.alpha
    gamma = config.gamma
    epsilon = config.q_learning.epsilon
    use_buffer = config.q_learning.use_buffer
    buffer_size = config.q_learning.buffer_size
    minibatch_size = config.q_learning.minibatch_size
    n_tiles = config.tc.num_tiles

    # Log cummulative reward, # of steps
    Envlogs = np.zeros((n, 2))

    pi = QLearningAgent(T, Q, epsilon, exploration_steps)

    if use_buffer:
        mem = [RingBuffer(capacity=buffer_size, dtype=(np.float, i)) for i in [
                env.observation_space.shape[0],
                1,
                1,
                env.observation_space.shape[0]
                ]
            ]

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        step_count = 0
        while not done:
            a = pi.action(s, i)
            s1, r, done, _ = env.step(a)

            # experience replay
            if use_buffer:
                mem[0].extendleft(np.array([s], dtype=np.float64))
                mem[1].extendleft(np.array([a], dtype=np.float64))
                mem[2].extendleft(np.array([r], dtype=np.float64))
                mem[3].extendleft(np.array([s1], dtype=np.float64))
                sample_idx = np.random.choice(
                    np.arange(len(mem[0])),
                    min(minibatch_size, len(mem[0]))
                )
                s_, a_, r_, s1_ = \
                    mem[0][sample_idx], np.int64(mem[1][sample_idx]), \
                    mem[2][sample_idx], mem[3][sample_idx]
            else:
                s_, a_, r_, s1_ = s, a, r, s1

            v = S(s1_, Q, T)
            delta = alpha * (r_ + (gamma * np.max(v, axis=-1)) - SA(s_, a_, Q, T))
            Q = update(Q, s_, a_, delta, T, n_tiles)
            pi.w = Q

            s = s1

            c_r += r
            step_count += 1

        Envlogs[i, 0], Envlogs[i, 1] = step_count, c_r
        # print('i: {}, s: {}, c: {}'.format(i, step_count, c_r))
    return Q, Envlogs


def DoubleQ(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    T: TileCoder
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: config
        Q: initial Q function
    ret:
        Q1: Q1 function; numpy array shape of [nS,nA]
        Q1: Q2 function; numpy array shape of [nS,nA]
        Qlogs1: Q1 logs
        Qlogs2: Q2 logs
        envlogs: (n, #s) logs
        visits: (n, #s, #a)
    """

    #####################
    # Double Q Learning (Hint: Sutton Book p. 135-136)
    #####################

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    alpha = config.dq_learning.alpha
    gamma = config.gamma
    epsilon = config.dq_learning.epsilon
    use_buffer = config.dq_learning.use_buffer
    buffer_size = config.dq_learning.buffer_size
    minibatch_size = config.dq_learning.minibatch_size
    n_tiles = config.tc.num_tiles

    Envlogs = np.zeros((n,2))

    Q1 = Q
    Q2 = copy.deepcopy(Q)

    # Keep track of which policy we are using
    # (see https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
    if np.random.random() < 0.5:
        pi = QLearningAgent(T, Q1, epsilon, exploration_steps)
        A = True
    else:
        pi = QLearningAgent(T, Q2, epsilon, exploration_steps)
        A = False
    if use_buffer:
        mem = [RingBuffer(capacity=buffer_size, dtype=(np.float, i)) for i in [
                env.observation_space.shape[0],
                1,
                1,
                env.observation_space.shape[0]
                ]
            ]

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        step_count = 0
        while not done:
            a = pi.action(s, i)

            s1, r, done, _ = env.step(a)

            # experience replay
            if use_buffer:
                mem[0].extendleft(np.array([s], dtype=np.float64))
                mem[1].extendleft(np.array([a], dtype=np.float64))
                mem[2].extendleft(np.array([r], dtype=np.float64))
                mem[3].extendleft(np.array([s1], dtype=np.float64))
                sample_idx = np.random.choice(
                    np.arange(len(mem[0])),
                    min(minibatch_size, len(mem[0]))
                )
                s_, a_, r_, s1_ = \
                    mem[0][sample_idx], np.int64(mem[1][sample_idx]), \
                    mem[2][sample_idx], mem[3][sample_idx]
            else:
                s_, a_, r_, s1_ = s, a, r, s1

            if A:
                aa = np.argmax(S(s1_, Q1, T), axis=-1)
                delta = alpha * (
                    r_ + gamma * SA(s1_, aa, Q2, T) - SA(s_, a_, Q1, T)
                )
            else:
                aa = np.argmax(S(s1_, Q2, T), axis=-1)
                delta = alpha * (
                    r_ + gamma * SA(s1_, aa, Q1, T) - SA(s_, a_, Q2, T)
                )

            s = s1

            if np.random.random() < 0.5:
                Q1 = update(Q1, s_, a_, delta, T, n_tiles)
                pi.w = Q1
                A = True
            else:
                Q1 = update(Q2, s_, a_, delta, T, n_tiles)
                pi.w = Q2
                A = False

            c_r += r
            step_count += 1

        Envlogs[i, 0], Envlogs[i, 1] = step_count, c_r

    return Q1, Q2, Envlogs 


def get_min_q(Q: np.array, max_estimators, active_estimators=-1):
    "Returns the minimum Q table"
    if active_estimators == -1:
        return np.amin(Q[:, :, :], axis=0).squeeze()
    else:
        ind = np.random.choice(np.arange(max_estimators), active_estimators)
        return np.amin(Q[ind, :, :], axis=0)


def MaxminQ(
    env: gym.Env,
    config: EasyDict,
    estimators: int,
    Q_init: np.array,
    T: TileCoder
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: config
        estimators: Number of Q tables
        Q_init: Q value initializer
    ret:
        Q: Q function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (n, #s) logs
        visits: (n, #s, #a)
    """

    #####################
    # Maxmin Q learning (ref. http://arxiv.org/abs/2002.06487)
    #####################

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    alpha = config.mmq_learning.alpha
    gamma = config.gamma
    epsilon = config.mmq_learning.epsilon
    buffer_size = config.mmq_learning.buffer_size
    minibatch_size = config.mmq_learning.minibatch_size
    n_tiles = config.tc.num_tiles

    Envlogs = np.zeros((n, 2))

    MMQ = np.repeat(Q_init[np.newaxis, :, :], estimators, axis=0)
    Q_min = get_min_q(MMQ, None)

    pi = QLearningAgent(T, Q_min, epsilon, exploration_steps)
    mem = [RingBuffer(capacity=buffer_size, dtype=(float, i)) for i in [
            env.observation_space.shape[0],
            1,
            1,
            env.observation_space.shape[0]
            ]
        ]

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        step_count = 0
        while not done:
            a = pi.action(s, i)

            s1, r, done, _ = env.step(a)

            update_ind = np.random.choice(estimators)
            mem[0].extendleft(np.array([s], dtype=np.float64))
            mem[1].extendleft(np.array([a], dtype=np.float64))
            mem[2].extendleft(np.array([r], dtype=np.float64))
            mem[3].extendleft(np.array([s1], dtype=np.float64))
            sample_idx = np.random.choice(
                np.arange(len(mem[0])),
                min(minibatch_size, len(mem[0]))
            )
            s_, a_, r_, s1_ = \
                mem[0][sample_idx], np.int64(mem[1][sample_idx]), \
                mem[2][sample_idx], mem[3][sample_idx]

            a_p = np.argmax(S(s1_, Q_min, T), axis=-1)
            delta = alpha * (r_ + gamma * SA(s1_, a_p, Q_min, T) - SA(s_, a_, MMQ[update_ind], T))

            s = s1

            MMQ[update_ind] = update(MMQ[update_ind], s_, a_, delta, T, n_tiles)
            Q_min = get_min_q(MMQ, None)
            pi.w = Q_min

            c_r += r
            step_count += 1

        Envlogs[i, 0], Envlogs[i, 1] = step_count, c_r
    return Q_min, Envlogs


def get_active_estimators(q_est, num_a_est, t):
    return np.argmax(q_est + 2 * np.sqrt(np.log(t)/(num_a_est + 1e-8))) + 1


def MaxminBanditQ(
    env: gym.Env,
    config: EasyDict,
    Q_init: np.array,
    T: TileCoder
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: config
        Q: initial Q function
    ret:
        Q: q_star function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (n, #s) logs
        visits: (n, #s, #a)
        bandits_logs: (n, #max_est, s)
    """

    #####################
    # Bandits to select # of estimators
    #####################

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    alpha = config.mmbq_learning.alpha
    gamma = config.gamma
    epsilon = config.mmbq_learning.epsilon
    max_estimators = config.mmbq_learning.max_estimators
    buffer_size = config.mmbq_learning.buffer_size
    reward_buffer = config.mmbq_learning.cum_len
    minibatch_size = config.mmbq_learning.minibatch_size
    bandit_lr = config.mmbq_learning.bandit_lr
    n_tiles = config.tc.num_tiles

    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)

    MMBQ = np.repeat(Q_init[np.newaxis, :, :], max_estimators, axis=0)

    q_est = np.zeros(max_estimators)
    num_a_est = np.zeros(max_estimators)

    active_estimators = get_active_estimators(q_est, num_a_est, 1)
    Q_min = get_min_q(MMBQ, max_estimators, active_estimators)

    pi = QLearningAgent(T, Q_min, epsilon, exploration_steps)

    mem = [RingBuffer(capacity=buffer_size, dtype=(np.float, i)) for i in [
            env.observation_space.shape[0],
            1,
            1,
            env.observation_space.shape[0]
            ]
        ]
    c_r_memory = deque([], maxlen = reward_buffer + 1)

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        step_count = 0
        while not done:
            a = pi.action(s, i)

            s1, r, done, _ = env.step(a)

            mem[0].extendleft(np.array([s], dtype=np.float64))
            mem[1].extendleft(np.array([a], dtype=np.float64))
            mem[2].extendleft(np.array([r], dtype=np.float64))
            mem[3].extendleft(np.array([s1], dtype=np.float64))
            sample_idx = np.random.choice(
                np.arange(len(mem[0])),
                min(minibatch_size, len(mem[0]))
            )
            update_ind = np.random.choice(max_estimators)
            s_, a_, r_, s1_ = \
                mem[0][sample_idx], np.int64(mem[1][sample_idx]), \
                mem[2][sample_idx], mem[3][sample_idx]

            a_p = np.argmax(S(s1_, Q_min, T), axis=-1)
            delta = alpha * (r_ + gamma * SA(s1_, a_p, Q_min, T) - SA(s_, a_, MMBQ[update_ind], T))
            MMBQ[update_ind] = update(MMBQ[update_ind], s_, a_, delta, T, n_tiles)

            Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
            pi.w = Q_min

            s = s1
            c_r += r
            step_count += 1

        c_r_memory.append([c_r, active_estimators])
        if len(c_r_memory) > reward_buffer:
            new_reward = (np.sum( list(c_r_memory), axis=0 )[0] - c_r_memory[0][0]) * 1.0 / reward_buffer
            new_reward /= 200.
            num_active_estimators = c_r_memory[0][1] - 1
            q_est[num_active_estimators] = q_est[num_active_estimators] + bandit_lr * (new_reward - q_est[num_active_estimators])
            num_a_est[ num_active_estimators ] += 1

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = q_est
        Bandit_logger[i, :, 1] = num_a_est

        Envlogs[i, 0], Envlogs[i, 1] = step_count, c_r
        # print('i: {}, s: {}, c: {}, q_est: {}, num: {}'.format(i, step_count, c_r, q_est, 2*np.sqrt(np.log(i+2)/(num_a_est + 1e-8))))
        active_estimators = get_active_estimators(q_est, num_a_est, i+2)

    return Q_min, Envlogs, Bandit_logger, Estimator_logger


def MaxminBanditQ_v2(
    env: gym.Env,
    config: EasyDict,
    Q_init: np.array,
    T: TileCoder
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: config
        Q: initial Q function
    ret:
        Q: q_star function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (n, #s) logs
        visits: (n, #s, #a)
        bandits_logs: (n, #max_est, s)
    """

    #####################
    # Bandits to select # of estimators
    #####################

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    alpha = config.mmbq_learning.alpha
    gamma = config.gamma
    epsilon = config.mmbq_learning.epsilon
    max_estimators = config.mmbq_learning.max_estimators
    buffer_size = config.mmbq_learning.buffer_size
    reward_buffer = config.mmbq_learning.cum_len
    minibatch_size = config.mmbq_learning.minibatch_size
    bandit_lr = config.mmbq_learning.bandit_lr
    n_tiles = config.tc.num_tiles

    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)

    MMBQ = np.repeat(Q_init[np.newaxis, :, :], max_estimators, axis=0)
    q_est = np.zeros(max_estimators)
    num_a_est = np.zeros(max_estimators)

    active_estimators = get_active_estimators(q_est, num_a_est, 1)
    Q_min = get_min_q(MMBQ, max_estimators, active_estimators)

    pi = QLearningAgent(T, Q_min, epsilon, exploration_steps)

    mem = [RingBuffer(capacity=buffer_size, dtype=(float, i)) for i in [
            env.observation_space.shape[0],
            1,
            1,
            env.observation_space.shape[0]
            ]
        ]
    c_r_memory = [deque([], maxlen = reward_buffer + 1) for _ in range(max_estimators+1)]

    for i in range( n ):
        s = env.reset()
        done = False
        c_r = 0
        step_count = 0
        while not done:
            a = pi.action(s, i)

            s1, r, done, _ = env.step(a)

            update_ind = np.random.choice(max_estimators)
            mem[0].extendleft(np.array([s], dtype=np.float64))
            mem[1].extendleft(np.array([a], dtype=np.float64))
            mem[2].extendleft(np.array([r], dtype=np.float64))
            mem[3].extendleft(np.array([s1], dtype=np.float64))
            sample_idx = np.random.choice(
                np.arange(len(mem[0])),
                min(minibatch_size, len(mem[0]))
            )
            s_, a_, r_, s1_ = \
                mem[0][sample_idx], np.int64(mem[1][sample_idx]), \
                mem[2][sample_idx], mem[3][sample_idx]

            a_p = np.argmax(S(s1_, Q_min, T), axis=-1)
            delta = alpha * (r_ + gamma * SA(s1_, a_p, Q_min, T) - SA(s_, a_, MMBQ[update_ind], T))
            MMBQ[update_ind] = update(MMBQ[update_ind], s_, a_, delta, T, n_tiles)

            Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
            pi.w = (Q_min)

            s = s1

            c_r += r
            step_count += 1

        c_r_memory[active_estimators].append([c_r, active_estimators])
        if len(c_r_memory[active_estimators]) > reward_buffer:
            new_reward = (np.sum( list(c_r_memory[active_estimators]), axis=0 )[0] - c_r_memory[active_estimators][0][0]) * 1.0 / reward_buffer
            new_reward /= 200.
            num_active_estimators = c_r_memory[active_estimators][0][1] - 1
            q_est[num_active_estimators] = q_est[num_active_estimators] + bandit_lr * (new_reward - q_est[num_active_estimators])
            num_a_est[ num_active_estimators ] += 1

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = q_est
        Bandit_logger[i, :, 1] = num_a_est

        Envlogs[i, 0], Envlogs[i, 1] = step_count, c_r
        # print('i: {}, s: {}, c: {}, q_est: {}, num: {}'.format(i, step_count, c_r, q_est, 10*np.sqrt(np.log(i+2)/(num_a_est + 1e-8))))
        active_estimators = get_active_estimators(q_est, num_a_est, i+2)

    return Q_min, Envlogs, Bandit_logger, Estimator_logger
