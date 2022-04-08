from typing import Tuple
import numpy as np

from policies.q_values import Policy, EGPolicy, GreedyPolicy
from tqdm import tqdm



def value_iteration(env, initV:np.array, theta:float, gamma:float, rng) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
        gamma: discount factor
        rng: rng
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
        q: q values, numpy array shape of [nS, nA]
    """

    #####################
    # Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    nS = initV.shape[0]
    assert nS == env.TD.shape[0], 'States in initV does not equal TD shape!'
    # (state, action, next_state)
    nA = env.TD.shape[1]
    while True:
        delta = 0
        for s_i in range(nS):
            v = initV[s_i]
            s = [0]*env.TD[s_i].shape[0]
            for a_i in range(env.TD[s_i].shape[0]):
                s[a_i] = np.sum(env.TD[s_i, a_i, :]*(env.R[s_i, a_i, :] + gamma*initV[:]))
            initV[s_i] = max(s)
            delta = max(delta, abs(v - initV[s_i]))
        if delta < theta:
            break

    V = initV

    # Compute Q values for policy
    Q = np.zeros((nS, nA))
    for s_i in range(nS):
        for a_i in range(nA):
            Q[s_i, a_i] = np.sum([
                    env.TD[s_i, a_i, s_p]*(env.R[s_i, a_i, s_p] + gamma*initV[s_p]) for s_p in range(nS)
                    ])

    pi = GreedyPolicy(Q, rng)

    return V, pi, Q


def Q_learning(env, n:int, alpha:float, gamma:float, epsilon:float, Q:np.array, rng) -> Tuple[np.array,Policy]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epslion: epsilon-greedy exploration
        Q: initial Q function
        rng:
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
    """

    #####################
    # Q Learning (Hint: Sutton Book p. 131)
    #####################

    terminal = env.final_state
    Q[terminal, :] = 0

    pi = EGPolicy(Q, epsilon, rng)

    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        while not done:
            a = pi.action(s)
            s1, r, done = env.step(a)

            Q[s,a] += alpha*(r + (gamma*np.max(Q[s1, :]) - Q[s, a]))
            s = s1
            pi.update(Q, epsilon)
        epsilon *= 0.995

    return Q


def DoubleQ(env, n:int, alpha:float, gamma:float, epsilon:float, Q1:np.array, Q2:np.array, rng) -> Tuple[np.array,Policy]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epsilon: epsilon-greedy exploration
        Q1: initial Q1 function
        Q2: initial Q2 function
        rng: rng
    ret:
        Q1: Q1 function; numpy array shape of [nS,nA]
        Q1: Q2 function; numpy array shape of [nS,nA]
    """

    #####################
    # Double Q Learning (Hint: Sutton Book p. 135-136)
    #####################


    assert Q1.all() == Q2.all(), 'Q values need to be different, otherwise there is no point in running Double Q learning...'

    terminal = env.final_state
    Q1[terminal, :] = 0
    Q2[terminal, :] = 0

    pi = EGPolicy(Q1, epsilon, rng)

    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        while not done:
            a = pi.action(s)
            s1, r, done = env.step(a)

            Q1[s,a] += alpha*(r + gamma*Q2[s1, np.argmax(Q1[s1, :])] - Q1[s, a])
            Q2[s,a] += alpha*(r + gamma*Q1[s1, np.argmax(Q2[s1, :])] - Q2[s, a])

            s = s1
            pi.update(Q1, epsilon)
        epsilon *= 0.95

    return Q1, Q2
