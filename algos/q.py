from typing import Tuple
import numpy as np
import gym
import copy

from collections import deque

from policies.q_values import EGPolicy
from easydict import EasyDict

from utils.buffers import RunningStats
from numpy_ringbuffer import RingBuffer

from utils.bandits import ExpWeights, Exp3, UCB


def get_min_q(Q: np.array, max_estimators, active_estimators=-1, fixed_estimators=False):
    """Returns the minimum Q table
    
    Args
        Q: Q table
        max_estimators: total number of estimators
        active_estimators: number of estimators currently used (if -1, use all estimators)
        fixed_estimators: If true, uses estimators [1, ..., active_estimators], if false, uses random set of
                            active_estimators
    """
    if active_estimators == -1:
        return np.amin(Q[:, :, :], axis=0).squeeze()
    else:
        if fixed_estimators:
            ind = np.arange(0, active_estimators)
        else:
            ind = np.random.choice(max_estimators, active_estimators)
        return np.amin( Q[ind, :, :], axis=0 ).squeeze()


def get_min_q_reorder(Q: np.array, max_estimators, active_estimators=-1, boost=0):
    """Returns the minimum Q table"""
    idx = [i for i in range(max_estimators)] # List of all indices

    # boost latest updated estimator to first position
    idx.remove(boost)
    idx = list((boost,)) + idx
    Q = Q[idx, :, :]

    ind = np.arange(0, active_estimators)
    return np.amin( Q[ind, :, :], axis=0).squeeze(), Q


def Q_learning(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    rng
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

    # Log Q value, cummulative reward, # of steps
    Qlogger = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n, 2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    terminal = env.final_state
    Q[terminal, :] = 0

    pi = EGPolicy(Q, epsilon, exploration_steps, rng)

    mem = deque([], maxlen=config.q_learning.buffer_size)

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1
            s1, r, done, _ = env.step(a)

            if use_buffer:
                mem.extendleft(np.array([[s, a, r, s1]]))
                if len(mem) == buffer_size:
                    sample_idx = np.random.choice(
                        np.arange(0, len(mem)),
                        minibatch_size
                    )
                    update_trans = mem[sample_idx]
                    s_, a_, r_, s1_ = \
                        np.int64(update_trans[:, 0]), \
                        np.int64(update_trans[:, 1]), \
                        update_trans[:, 2], \
                        np.int64(update_trans[:, 3])
            else:
                s_, a_, r_, s1_ = s, a, r, s1

                Q[ np.int64(s_), np.int64(a_) ] += alpha * (r_ + (gamma * np.max(Q[ np.int64(s1_), :])) - Q[ np.int64(s_), np.int64(a_) ])

            s = s1

            pi.update(Q)

            c_r += r

        Qlogger[i, ::] = Q
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return Q, Qlogger, Envlogs, VisitLogger


def DoubleQ(
    env: gym.Env,
    config: EasyDict,
    Q: np.ndarray,
    rng
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

    # Log Q value, cummulative reward, # of steps
    Qlogger1 = np.zeros((n, env.nS, env.nA,))
    Qlogger2 = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n,2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    terminal = env.final_state
    Q1 = Q
    Q2 = copy.deepcopy(Q)
    Q1[terminal, :] = 0
    Q2[terminal, :] = 0

    # Keep track of which policy we are using
    # (see https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
    if np.random.rand() < 0.5:
        pi = EGPolicy(Q1, epsilon, exploration_steps, rng)
        A = True
    else:
        pi = EGPolicy(Q2, epsilon, exploration_steps, rng)
        A = False

    mem = deque([], maxlen=config.dq_learning.buffer_size)

    for i in range(n):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)
            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)

            # experience replay
            if use_buffer:
                mem.extendleft(np.array([[s, a, r, s1]]))
                if len(mem) == buffer_size:
                    sample_idx = np.random.choice(
                        np.arange(0, len(mem)),
                        minibatch_size
                    )
                    update_trans = mem[sample_idx]
                    s_, a_, r_, s1_ = \
                            np.int64(update_trans[:, 0]), \
                            np.int64(update_trans[:, 1]), \
                            update_trans[:, 2], \
                            np.int64(update_trans[:, 3])

                    if A:
                        Q[0, np.int64(s_), np.int64(a_)] += alpha * (
                            r_ + gamma * Q[1, np.int64(s1_), np.argmax(Q[0, np.int64(s1_), :], axis=-1)] - Q[0, np.int64(s_), np.int64(a_)]
                        )
                    else:
                        Q[1, np.int64(s_), np.int64(a_)] += alpha * (
                            r_ + gamma * Q[0, np.int64(s1_), np.argmax(Q[1, np.int64(s1_), :], axis=-1)] - Q[1, np.int64(s_), np.int64(a_) ]
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

    return Q1, Q2, Qlogger1, Qlogger2, Envlogs, VisitLogger


def MaxminQ(
    env: gym.Env,
    config: EasyDict,
    estimators: int,
    Q_init: np.array,
    rng
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

    MMQ_logger = np.zeros((n, env.nS, env.nA,))
    Envlogs = np.zeros((n, 2))
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMQ = np.repeat(Q_init[np.newaxis, :, :], estimators, axis=0)
    assert MMQ.shape == (estimators, env.nS, env.nA)

    terminal = env.final_state
    MMQ[:, terminal, :] = 0

    Q_min = get_min_q(MMQ, None)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)

    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))

    memory = deque([], maxlen=config.mmq_learning.buffer_size)

    for i in range( config.mmq_learning.steps ):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)
            memory.append([s, a, r, s1])

            if len(mem) == buffer_size:
                update_est_idx = np.random.choice(estimators)
                sample_idx = np.random.choice(
                    np.arange(0, len(mem)),
                    minibatch_size
                )
                update_trans = mem[sample_idx]
                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)

                MMQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])] += \
                        alpha \
                        * (update_trans[:, 2] \
                            + gamma \
                            * Q_min[np.int64(update_trans[:, 3]), a_p] \
                            - MMQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])]
                        )

            s = s1

            Q_min = get_min_q(MMQ, None)
            pi.update(Q_min)

            c_r += r

        MMQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

    return Q_min, MMQ_logger, Envlogs, VisitLogger


def MaxminBanditQ(
    env: gym.Env,
    config: EasyDict,
    Q_init: np.array,
    rng
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
    bandit_r_buf_len = config.mmbq_learning.cum_len
    minibatch_size = config.mmbq_learning.minibatch_size
    bandit_lr = config.mmbq_learning.bandit_lr

    MMBQ_logger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMBQ = np.repeat(Q_init[np.newaxis, :, :], max_estimators, axis=0)
    assert MMBQ.shape == (max_estimators, env.nS, env.nA)

    TDC = UCB(max_estimators, bandit_lr)

    num_a_est = np.zeros(max_estimators)

    terminal = env.final_state
    MMBQ[:, terminal, :] = 0

    active_estimators = TDC.sample()

    Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)

    # memory = RingBuffer(capacity=config.mmbq_learning.buffer_size, dtype=(float, (4)))
    mem = deque([], maxlen = config.mmbq_learning.buffer_size)
    c_r_memory = deque([], maxlen = config.mmbq_learning.cum_len + 1)

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
                update_est_idx = np.random.choice(np.arange(0, max_estimators), max_estimators - active_estimators + 1)
                sample_idx = np.random.choice(
                    np.arange(0, len(mem)),
                    minibatch_size
                )
                update_trans = mem[sample_idx]
                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)
                MMBQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])] += \
                        alpha \
                        * (update_trans[:, 2] \
                            + gamma \
                            * Q_min[np.int64(update_trans[:, 3]), a_p] \
                            - MMBQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])]
                        )

            s = s1

            Q_min = get_min_q(MMBQ, max_estimators, active_estimators)
            pi.update(Q_min)

            c_r += r

        c_r_memory.append([c_r, active_estimators])
        if len(c_r_memory) > bandit_r_buf_len:
            bandit_rew = (np.sum(list(c_r_memory), axis=0)[0] - (c_r_memory[0][0]))/bandit_r_buf_len
            num_active_estimators = c_r_memory[-1][1] - 1
            num_a_est[ num_active_estimators ] += 1

            # Change based on env, used to normalize the cumulative reward
            TDC.update((bandit_rew+1.5)/3)

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = TDC.get_values()
        Bandit_logger[i, :, 1] = num_a_est

        MMBQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

        active_estimators = TDC.sample()

    return Q_min, MMBQ_logger, Envlogs, VisitLogger, Bandit_logger, Estimator_logger


def MaxminBanditQ_v2(
    env: gym.Env,
    config: EasyDict,
    Q_init: np.array,
    rng
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
    bandit_r_buf_len = config.mmbq_learning.cum_len
    minibatch_size = config.mmbq_learning.minibatch_size

    MMBQ_logger = np.empty((n, env.nS, env.nA,))
    Envlogs = np.empty((n , 2))
    Bandit_logger = np.zeros((n, max_estimators, 2))
    Estimator_logger = np.zeros((n,), dtype=np.int64)
    VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

    MMBQ = np.repeat(Q_init[np.newaxis, :, :], max_estimators, axis=0)
    assert MMBQ.shape == (max_estimators, env.nS, env.nA)

    TDC = Exp3(n_arms=max_estimators, gamma=0.15)
    # TDC = ExpWeights(arms=[i for i in range(1, max_estimators+1)], lr=0.2, window=5, use_std=False)

    num_a_est = np.zeros(max_estimators)

    terminal = env.final_state
    MMBQ[:, terminal, :] = 0

    active_estimators = TDC.sample()

    Q_min = get_min_q(MMBQ, max_estimators, active_estimators, fixed_estimators=True)
    assert Q_min.shape == (env.nS, env.nA)

    pi = EGPolicy(Q_min, epsilon, exploration_steps, rng)

    mem = RingBuffer(capacity=buffer_size, dtype=(float, (4)))
    c_r_memory = [deque([], maxlen = bandit_r_buf_len + 1) for _ in range(max_estimators+1)]

    for i in range( n ):
        s = env.reset()
        done = False
        c_r = 0
        while not done:
            a = pi.action(s, i)

            VisitLogger[i, s, a] += 1

            s1, r, done, _ = env.step(a)

            mem.extendleft(np.array([[s, a, r, s1]]))

            update_est_idx = 0
            if len(mem) == buffer_size:
                update_est_idx = np.random.choice(np.arange(0, max_estimators))
                sample_idx = np.random.choice(
                    np.arange(0, len(mem)),
                    minibatch_size
                )
                update_trans = mem[sample_idx]
                a_p = np.argmax(Q_min[np.int64(update_trans[:, 3]), :], axis=-1)

                td_error = update_trans[:, 2] \
                            + gamma \
                            * Q_min[np.int64(update_trans[:, 3]), a_p] \
                            - MMBQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])]

                MMBQ[update_est_idx, np.int64(update_trans[:, 0]), np.int64(update_trans[:, 1])] += \
                        alpha \
                        * (td_error)

            s = s1

            Q_min, MMBQ = get_min_q_reorder(MMBQ, max_estimators, active_estimators, update_est_idx)
            pi.update(Q_min)

            c_r += r

        # Change based on env, used to normalize the cumulative reward
        TDC.update((c_r+1.5)/3)

        c_r_memory[active_estimators].append([c_r, active_estimators])
        if len(c_r_memory[active_estimators]) > bandit_r_buf_len:
            num_active_estimators = c_r_memory[active_estimators][-1][1] - 1
            num_a_est[ num_active_estimators ] += 1

        Estimator_logger[i] = active_estimators
        Bandit_logger[i, :, 0] = TDC.get_values()
        Bandit_logger[i, :, 1] = num_a_est

        MMBQ_logger[i, :, :] = Q_min
        Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

        if i != n - 1:
            active_estimators = TDC.sample()
            Q_min, MMBQ = get_min_q_reorder(MMBQ, max_estimators, active_estimators, boost=update_est_idx)
            pi.update(Q_min)


    return Q_min, MMBQ_logger, Envlogs, VisitLogger, Bandit_logger, Estimator_logger


# def PessimisticQ(
#     env: gym.Env,
#     config: EasyDict,
#     PQ: np.array,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     input:
#         env: environment
#         config: config
#         PQ: initial Q function
#     ret:
#         Q: $q_star$ function; numpy array shape of [nS,nA]
#         Qlogs: Q logs
#         envlogs: (n, #s) logs
#         visits: (n, #s, #a)
#     """

#     #####################
#     # Pessimistic Q learning (ref. https://arxiv.org/pdf/2202.13890.pdf)
#     #####################

#     n = config.exp.steps
#     exploration_steps = config.exp.exploration_steps
#     alpha = config.pessimistic_q_learning.alpha
#     gamma = config.gamma
#     epsilon = config.pessimistic_q_learning.epsilon
#     pessimism_coeff = config.pessimistic_q_learning.pessimism_coeff

#     # Log Q value, cummulative reward, # of steps
#     Qlogger = np.zeros((n, env.nS, env.nA,))
#     Envlogs = np.zeros((n,2))
#     VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

#     pi = EGPolicy(PQ, epsilon, exploration_steps)
#     visit = np.zeros(PQ.shape) + 1

#     for i in range(n):
#         s = env.reset()
#         done = False
#         c_r = 0
#         while not done:
#             a = pi.action(s, i)
#             VisitLogger[i, s, a] += 1

#             s1, r, done, _ = env.step(a)

#             visit[s, a] += 1
#             PQ[s, a] += alpha * (
#                 r
#                 + gamma * np.max(PQ[s1, :])
#                 - PQ[s, a]
#                 - (pessimism_coeff / visit[s, a])
#             )
#             s = s1
#             pi.update(PQ)

#             c_r += r

#         Qlogger[i, ::] = PQ
#         Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

#     return PQ, Qlogger, np.int64(Envlogs), VisitLogger


# def MeanVarianceQ(
#     env: gym.Env,
#     config: EasyDict,
#     MVQ: np.array,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     input:
#         env: environment
#         config: config
#         Q: initial Q function
#     ret:
#         Q: q_star function; numpy array shape of [nS,nA]
#         Qlogs: Q logs
#         envlogs: (n, #s) logs
#         visits: (n, #s, #a)
#     """

#     #####################
#     # Keep track of mean and variance rewards
#     #####################

#     n = config.exp.steps
#     exploration_steps = config.exp.exploration_steps
#     alpha = config.meanvar_q_learning.alpha
#     gamma = config.gamma
#     epsilon = config.meanvar_q_learning.epsilon
#     coeff = config.meanvar_q_learning.coeff

#     # Log Q value, cummulative reward, # of steps
#     Qlogger = np.zeros((n, env.nS, env.nA,))
#     Envlogs = np.zeros((n,2))
#     VisitLogger = np.zeros((n, env.nS, env.nA), dtype=np.int64)

#     pi = EGPolicy(MVQ, epsilon, exploration_steps)

#     # track mean and variance of reward
#     rs = [[RunningStats(20) for i in range(env.nA)] for k in range(env.nS)]

#     for i in range(n):
#         s = env.reset()
#         done = False
#         c_r = 0
#         while not done:
#             a = pi.action(s)

#             VisitLogger[i, s, a] += 1

#             s1, r, done, _ = env.step(a)

#             rs[s][a].push(r)

#             MVQ[s, a] += alpha * (
#                 r
#                 + gamma * np.max(MVQ[s1, :])
#                 - MVQ[s, a]
#                 - (0.1 * rs[s][a].n * rs[s][a].get_var())
#             )
#             s = s1

#             c_r += r

#         Qlogger[i, ::] = MVQ
#         Envlogs[i, 0], Envlogs[i, 1] = c_r, env.step_count

#     del rs
#     return MVQ, Qlogger, Envlogs, VisitLogger
