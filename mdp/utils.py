import sys
import numpy as np
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

def getValueFromDict(indict, key, defaultVal=None):
  if key in indict.keys():
    return indict[key]
  else:
    return defaultVal

def argmax(x):
  return np.random.choice(np.nonzero(x==np.max(x))[0])
  
def normalize_sum(x, **kwargs):
  x_norm = x + np.min(x)
  if np.isclose(np.sum(x_norm),0):
    x_norm = x_norm + (1.0/len(x))
  else:
    x_norm = x_norm/np.sum(x_norm)
  return x_norm
  
def normalize_softmax_nonsafe(x, **kwargs):
  return np.exp(x) / np.sum(np.exp(x))
  
def normalize_softmax(x, **kwargs):
  shiftx = x - np.max(x)
  exps = np.exp(shiftx)
  return exps / np.sum(exps)
  
def normalize_greedy(x, **kwargs):
  argmax_function=kwargs["argmaxfun"]
  x_norm = np.zeros_like(x)
  x_norm[argmax_function(x)] = 1.0
  return x_norm

def normalize_esoft(x, **kwargs):
  argmax_function = kwargs["argmaxfun"]
  epsilon = kwargs["epsilon"]
  x_norm = np.zeros_like(x) + epsilon/(len(x) - 1)
  x_norm[argmax_function(x)] = 1.0 - epsilon
  return x_norm

def mapValues(val, minSrc, maxSrc, minDest, maxDest):
  aux = (val - minSrc)/(maxSrc-minSrc)
  return aux*(maxDest-minDest) + minDest

def selectAction_egreedy(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax) 
  epsilon = kwargs["epsilon"]
  if np.random.rand()<epsilon:
    action = np.random.randint(0, len(actionValues))
  else:
    action = argmax_function(actionValues)
  return action

def selectAction_greedy(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  action = argmax_function(actionValues)
  return action

def selectAction_esoft(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  epsilon = kwargs["epsilon"]
  p = np.zeros_like(actionValues) + epsilon/(len(actionValues) - 1)
  p[argmax_function(actionValues)] = 1.0 - epsilon
  return np.random.choice(len(p), p=p)
  
def selectAction_esoft_(actionValues, **kwargs):
  # TODO: consider implementing this
  epsilon = kwargs["epsilon"]
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  q_max = np.max(actionValues) 
  n_greedy_actions = 0
  greedy_actions = []
  for i in range(len(actionValues)): 
    if actionValues[i] == q_max: 
      n_greedy_actions += 1
      greedy_actions.append(i)
  non_greedy_action_probability = epsilon / len(actionValues)
  greedy_action_probability = ((1.0 - epsilon) / n_greedy_actions) + non_greedy_action_probability 
  p=np.zeros(len(actionValues))+non_greedy_action_probability
  p[greedy_actions] = greedy_action_probability
  return np.random.choice(len(p), p=p)
  
def selectAction_softmax(actionValues, **kwargs):
  p = normalize_softmax(actionValues)
  return np.random.choice(len(p), p=p)
  
def selectAction_UCB(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax) 
  c = kwargs["c"]
  t = kwargs["t"]
  N = kwargs["N"]
  if np.min(N)==0:
    return np.argmin(N)
  else:
    return argmax_function(actionValues + c*np.sqrt(np.log(t)/N))
