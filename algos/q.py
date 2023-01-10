from typing import Tuple
import copy, os, time
from collections import deque

import gym
import numpy as np
from numpy_ringbuffer import RingBuffer

from policies.q_values import EGPolicy
from easydict import EasyDict
from utils.bandits import ExpWeights, Exp3, UCB, ThompsonSamplingNonStationary


BANDIT_ALGO = {
    "ucb": UCB,
    "exp3": Exp3,
    "expw": ExpWeights,
    "thompson": ThompsonSamplingNonStationary,
}


def get_min_q(
    Q: np.ndarray,
    max_estimators: int,
    active_estimators: int = -1,
    fixed_estimators: bool = False
) -> np.ndarray:
    """Returns the minimum Q table

    Args
        Q: Q table
        max_estimators: total number of estimators
        active_estimators: number of estimators currently used (if -1, use all estimators)
        fixed_estimators: If true, uses estimators [-, ..., active_estimators], if false, uses random set of
                            active_estimators
    """
    if active_estimators == -1:
        return np.amin(Q[:, :, :], axis=0).squeeze()
    else:
        if fixed_estimators:
            ind = np.arange(0, active_estimators)
        else:
            ind = np.random.choice(max_estimators, active_estimators)
        return np.amin(Q[ind, :, :], axis=0).squeeze()


def Q_learning(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    rng: np.random.Generator,
    alpha: float
    ):
    """
    input:
        env: environment
        config: configs
        Q: initial Q function
        rng: Random Generator
        alpha: learning rate
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: reward logs
        visits: state visit logs
    """

    #####################
    # Q Learning (Sutton Book p. 131)
    #####################

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    gamma = config.gamma
    epsilon = config.q_learning.epsilon
    buffer_size = config.q_learning.buffer_size
    minibatch_size = config.q_learning.minibatch_size

    # Log Q value, cummulative reward, # of steps
    Qlogger = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n, 2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    terminal = env.final_state
    Q[terminal, :] = 0

    pi = EGPolicy(Q, epsilon, exploration_steps, rng)
    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1
            s1, r, done, _ = env.step(a)

            mem.append([s, a, r, s1])

            if len(mem) == buffer_size:
                sample_idx = np.random.choice(np.arange(0, len(mem)), minibatch_size)
                update_trans = mem[sample_idx]
                s_, a_, r_, s1_ = (
                    np.int64(update_trans[:, 0]),
                    np.int64(update_trans[:, 1]),
                    update_trans[:, 2],
                    np.int64(update_trans[:, 3]),
                )

                Q[np.int64(s_), np.int64(a_)] += alpha * (
                    r_
                    + (gamma * Q[np.int64(s1_), np.argmax(Q[np.int64(s1_), :], axis=-1)])
                    - Q[np.int64(s_), np.int64(a_)]
                )

            s = s1
            c_r += r

            pi.update(Q)

        Qlogger[i, ::] = Q
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return {"Q": Q, "QLog": Qlogger, "EnvLog": Envlogs, "VisitLog": VisitLogger}


def DoubleQ(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    rng: np.random.Generator,
    alpha: float
    ):
    """
    input:
        env: environment
        config: configs
        Q: initial Q function
        rng: Random Generator
        alpha: learning rate
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

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    gamma = config.gamma
    epsilon = config.dq_learning.epsilon
    buffer_size = config.dq_learning.buffer_size
    minibatch_size = config.dq_learning.minibatch_size

    # Log Q value, cummulative reward, # of steps
    Qlogger1 = np.zeros((n, env.nS, env.nA,))
    Qlogger2 = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n,2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    terminal = env.final_state
    Q[terminal, :] = 0
    Q1 = Q
    Q2 = copy.deepcopy(Q)

    # Keep track of which policy we are using
    # (see https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
    if np.random.rand() <= 0.5:
        pi = EGPolicy(Q1, epsilon, exploration_steps, rng)
        A = True
    else:
        pi = EGPolicy(Q2, epsilon, exploration_steps, rng)
        A = False

    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)
            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)
            mem.append([s, a, r, s1])

            if len(mem) == buffer_size:
                sample_idx = np.random.choice(np.arange(0, len(mem)), minibatch_size)
                update_trans = mem[sample_idx]
                s_, a_, r_, s1_ = (
                    np.int64(update_trans[:, 0]),
                    np.int64(update_trans[:, 1]),
                    update_trans[:, 2],
                    np.int64(update_trans[:, 3]),
                )

                if A:
                    Q1[np.int64(s_), np.int64(a_)] += alpha * (
                        r_
                        + (gamma * Q2[np.int64(s1_), np.argmax(Q1[np.int64(s1_), :], axis=-1)])
                        - Q1[np.int64(s_), np.int64(a_)]
                    )
                else:
                    Q2[np.int64(s_), np.int64(a_)] += alpha * (
                        r_
                        + (gamma * Q1[np.int64(s1_), np.argmax(Q2[np.int64(s1_), :], axis=-1)])
                        - Q2[np.int64(s_), np.int64(a_)]
                    )

            s = s1

            if np.random.random() < 0.5:
                pi.update(Q1)
                A = True
            else:
                pi.update(Q2)
                A = False

            c_r += r

        Qlogger1[i, :, :] = Q1[:, :]
        Qlogger2[i, :, :] = Q2[:, :]
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return {
        "Q1": Q1,
        "Q2": Q2,
        "QLog": Qlogger1,
        "Qlog2": Qlogger2,
        "EnvLog": Envlogs,
        "VisitLog": VisitLogger,
    }


def MaxminQ(
    env: gym.Env,
    config: EasyDict,
    estimators: int,
    Q: np.array,
    rng: np.random.Generator,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: configs
        estimators: number of estimators (N)
        Q: initial Q function
        rng: Random Generator
        alpha: learning rate
    ret:
        Q: Q function; numpy array shape of [nS,nA]
        Qlogs: Q logs
        envlogs: (n, #s) logs
        visits: (n, #s, #a)
    """

    #####################
    # Maxmin Q learning (ref. http://arxiv.org/abs/2002.06487)
    #####################

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    gamma = config.gamma
    epsilon = config.mmq_learning.epsilon
    buffer_size = config.mmq_learning.buffer_size
    minibatch_size = config.mmq_learning.minibatch_size

    MMQ_logger = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n, 2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMQ = np.repeat(Q[np.newaxis, :, :], estimators, axis=0)
    assert MMQ.shape == (estimators, env.nS, env.nA)

    terminal = env.final_state
    MMQ[:, terminal, :] = 0

    Q_min = get_min_q(MMQ, None)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)
    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)
            mem.append([s, a, r, s1])

            if len(mem) == buffer_size:
                # pick a random estimator count (N)
                update_est_idx = np.random.choice(estimators)

                # Sample minibatch
                sample_idx = np.random.choice(np.arange(0, len(mem)), minibatch_size)
                update_trans = mem[sample_idx]

                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)

                MMQ[
                    update_est_idx,
                    np.int64(update_trans[:, 0]),
                    np.int64(update_trans[:, 1]),
                ] += alpha * (
                    update_trans[:, 2]
                    + (gamma * Q_min[np.int64(update_trans[:, 3]), a_p])
                    - MMQ[
                        update_est_idx,
                        np.int64(update_trans[:, 0]),
                        np.int64(update_trans[:, 1]),
                    ]
                )

            s = s1

            Q_min = get_min_q(MMQ, max_estimators=estimators, active_estimators=-1)
            pi.update(Q_min)

            c_r += r

        MMQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return {"Q": Q_min, "QLog": MMQ_logger, "EnvLog": Envlogs, "VisitLog": VisitLogger}


def MaxminBanditQ(
    env: gym.Env,
    config: EasyDict,
    Q: np.array,
    rng: np.random.Generator,
    alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: configs
        Q: initial Q function
        rng: Random Generator
        alpha: learning rate
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

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    gamma = config.gamma
    epsilon = config.mmbq_learning.epsilon
    max_estimators = config.mmbq_learning.max_estimators
    buffer_size = config.mmbq_learning.buffer_size
    bandit_r_buf_len = config.mmbq_learning.cum_len
    minibatch_size = config.mmbq_learning.minibatch_size

    MMBQ_logger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMBQ = np.repeat(Q[np.newaxis, :, :], max_estimators, axis=0)
    assert MMBQ.shape == (max_estimators, env.nS, env.nA)

    args = dict(config.mmbq_learning.algo_params)
    args['N'] = max_estimators
    TDC = BANDIT_ALGO[config.mmbq_learning.algo](**args)

    num_a_est = np.zeros(max_estimators)

    terminal = env.final_state
    MMBQ[:, terminal, :] = 0

    active_estimators = TDC.sample()

    Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)

    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))
    c_r_memory = deque([], maxlen=bandit_r_buf_len + 1)

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)

            mem.append([s, a, r, s1])

            if len(mem) == buffer_size:
                sample_idx = np.random.choice(np.arange(0, len(mem)), minibatch_size)
                update_est_idx = np.random.choice(max_estimators)
                update_trans = mem[sample_idx]
                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)
                MMBQ[
                    update_est_idx,
                    np.int64(update_trans[:, 0]),
                    np.int64(update_trans[:, 1]),
                ] += alpha * (
                    update_trans[:, 2]
                    + (gamma * Q_min[np.int64(update_trans[:, 3]), a_p])
                    - MMBQ[
                        update_est_idx,
                        np.int64(update_trans[:, 0]),
                        np.int64(update_trans[:, 1]),
                    ]
                )

            s = s1

            Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
            pi.update(Q_min)

            c_r += r

        c_r_memory.append([c_r, active_estimators])
        if len(c_r_memory) > bandit_r_buf_len:
            bandit_rew = (
                np.sum(list(c_r_memory), axis=0)[0] - (c_r_memory[0][0])
            ) / bandit_r_buf_len
            num_active_estimators = c_r_memory[-1][1] - 1
            num_a_est[num_active_estimators] += 1

            # Change based on env, used to normalize the cumulative reward
            TDC.update(bandit_rew + 1)

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = TDC.get_values()
        Bandit_logger[i, :, 1] = num_a_est

        MMBQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

        active_estimators = TDC.sample()

    return {
        "Q": Q_min,
        "QLog": MMBQ_logger,
        "EnvLog": Envlogs,
        "VisitLog": VisitLogger,
        "BanditLog": Bandit_logger,
        "EstimLog": Estimator_logger,
    }


def MaxminBanditQ_v2(
    env: gym.Env,
    config: EasyDict,
    Q: np.array,
    rng: np.random.Generator,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    input:
        env: environment
        config: configs
        Q: initial Q function
        rng: Random Generator
        alpha: learning rate
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

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n = config.exp.steps
    exploration_steps = config.exp.exploration_steps
    gamma = config.gamma
    epsilon = config.mmbq2_learning.epsilon
    max_estimators = config.mmbq2_learning.max_estimators
    buffer_size = config.mmbq2_learning.buffer_size
    bandit_r_buf_len = config.mmbq2_learning.cum_len
    minibatch_size = config.mmbq2_learning.minibatch_size

    MMBQ_logger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMBQ = np.repeat(Q[np.newaxis, :, :], max_estimators, axis=0)
    assert MMBQ.shape == (max_estimators, env.nS, env.nA)

    args = dict(config.mmbq2_learning.algo_params)
    args['N'] = max_estimators
    TDC = BANDIT_ALGO[config.mmbq2_learning.algo](**args)

    num_a_est = np.zeros(max_estimators)  # Track number of times estimator is selected
    est_upd_age = np.zeros(max_estimators)  # Track time since last update to estimator
    idxs = np.arange(0, max_estimators)  # Default indexing of estimators

    terminal = env.final_state
    MMBQ[:, terminal, :] = 0

    active_estimators = TDC.sample()

    Q_min = get_min_q(MMBQ, max_estimators, active_estimators, fixed_estimators=True)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)

    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))
    c_r_memory = [
        deque([], maxlen=bandit_r_buf_len + 1) for _ in range(max_estimators + 1)
    ]

    age = 0
    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)

            mem.append([s, a, r, s1])

            if len(mem) == buffer_size:
                update_ind = np.random.choice(max_estimators, minibatch_size)
                est_upd_age += np.bincount(
                    update_ind, minlength=max_estimators
                )  # update estimators' age
                idxs = np.argsort(est_upd_age)[::-1] # sort in decending order

                sample_idx = np.random.choice(np.arange(0, len(mem)), minibatch_size)

                update_trans = mem[sample_idx]
                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)

                MMBQ[
                    update_ind,
                    np.int64(update_trans[:, 0]),
                    np.int64(update_trans[:, 1]),
                ] += alpha * (
                    update_trans[:, 2]
                    + (gamma * Q_min[np.int64(update_trans[:, 3]), a_p])
                    - MMBQ[
                        update_ind,
                        np.int64(update_trans[:, 0]),
                        np.int64(update_trans[:, 1]),
                    ]
                )

            s = s1

            c_r += r
            age += 1

            MMBQ = MMBQ[idxs, :, :]
            Q_min = get_min_q(MMBQ, max_estimators, active_estimators, fixed_estimators=True)

            pi.update(Q_min)

        # Change based on env, used to normalize the cumulative reward
        TDC.update(c_r + 1)  # (np.clip(c_r, -1, 1)+1)/2)

        c_r_memory[active_estimators].append([c_r, active_estimators])
        if len(c_r_memory[active_estimators]) > bandit_r_buf_len:
            num_active_estimators = c_r_memory[active_estimators][-1][1] - 1
            num_a_est[num_active_estimators] += 1

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = TDC.get_values()
        Bandit_logger[i, :, 1] = num_a_est

        MMBQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

        if i != n - 1:
            active_estimators = TDC.sample()
            MMBQ = MMBQ[idxs, :, :]
            Q_min = get_min_q(MMBQ, max_estimators, active_estimators, fixed_estimators=True)
            pi.update(Q_min)

    return {
        "Q": Q_min,
        "QLog": MMBQ_logger,
        "EnvLog": Envlogs,
        "VisitLog": VisitLogger,
        "BanditLog": Bandit_logger,
        "EstimLog": Estimator_logger,
    }
