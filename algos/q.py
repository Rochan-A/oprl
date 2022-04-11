from typing import Tuple
import numpy as np

from policies.q_values import Policy, EGPolicy
from tqdm import tqdm


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
    vl = []
    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        sl = []
        al = []
        ql = []
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)

            sl.append(s)
            al.append(a)
            ql.append( np.max(Q[s, :]) )
            Q[s,a] += alpha*(r + (gamma*np.max(Q[s1, :]) - Q[s, a]))
            s = s1
            pi.update(Q, epsilon)
        epsilon *= 0.95
        vl.append([sl, al, ql])

    return Q, vl


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

    terminal = env.final_state
    Q1[terminal, :] = 0
    Q2[terminal, :] = 0

    pi = EGPolicy(Q1, epsilon, rng)
    vl = []

    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        sl = []
        al = []
        ql1 = []
        ql2 = []
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)

            sl.append(s)
            al.append(a)
            ql1.append( np.max(Q1[s, :]) )
            ql2.append( np.max(Q2[s, :]) )

            if rng.random() < 0.5:
                Q1[s,a] += alpha*(r + gamma*Q2[s1, np.argmax(Q1[s1, :])] - Q1[s, a])
            else:
                Q2[s,a] += alpha*(r + gamma*Q1[s1, np.argmax(Q2[s1, :])] - Q2[s, a])

            s = s1
            pi.update(Q1, epsilon)
        epsilon *= 0.95
        vl.append([sl, al, ql1, ql2])

    return Q1, Q2, vl


def PessimisticQ(env, n:int, alpha:float, gamma:float, epsilon:float, pessimism_coeff:float, PQ:np.array, rng):

    pi = EGPolicy(PQ, epsilon, rng)

    visit = np.zeros(PQ.shape) + 1
    vl = []

    for i in tqdm(range(n)):
        s = env.reset()
        sl = []
        al = []
        ql = []
        done = False
        while not done:
            a = pi.action(s)
            s1, r, done, _ = env.step(a)
            sl.append(s)
            al.append(a)
            ql.append( np.max(PQ[s, :]) )

            visit[s, a] += 1
            PQ[s,a] += alpha*(r + gamma*np.max(PQ[s1, :]) - PQ[s, a] - (pessimism_coeff / visit[s, a]) )
            s = s1
            pi.update(PQ, epsilon)
        epsilon *= 0.95
        vl.append([sl, al, ql])

    return PQ, vl
