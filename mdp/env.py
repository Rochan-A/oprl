import numpy as np

class MaximizationBias():

  N_STATES = 4
  STATE_B = 1
  STATE_A = 2
  STATE_TERMINAL_R = 3
  STATE_TERMINAL_L = 0
  N_ACTIONS_MIN = 2
  ACTION_LEFT = 1
  ACTION_RIGHT = 0
  
  def __init__(self, mu=-0.1, std=1, nActions=10):
    self.mu = mu
    self.std = std
    #self.nActions = max(self.N_ACTIONS_MIN, nActions)
    self.nActions = self.N_ACTIONS_MIN + nActions
    # States
    self.nStates = self.N_STATES
    self.startState = self.STATE_A
    self.terminalStates = [self.STATE_TERMINAL_L, self.STATE_TERMINAL_R]
    self.stateTransitions = {
      self.STATE_A: {self.ACTION_RIGHT: self.STATE_TERMINAL_R, self.ACTION_LEFT: self.STATE_B},
      self.STATE_B: {a: self.STATE_TERMINAL_L for a in range(self.N_ACTIONS_MIN, self.N_ACTIONS_MIN+self.nActions)},
      self.STATE_TERMINAL_R: {self.ACTION_RIGHT: self.STATE_TERMINAL_R, self.ACTION_LEFT: self.STATE_TERMINAL_R},
      self.STATE_TERMINAL_L: {self.ACTION_LEFT: self.STATE_TERMINAL_L, self.ACTION_RIGHT: self.STATE_TERMINAL_L}
    }
    # Actions
    self.stateActionMapping = {
      self.STATE_A: [self.ACTION_RIGHT, self.ACTION_LEFT],
      #self.STATE_B: [i for i in range(self.nActions)],
      self.STATE_B: [i for i in range(self.N_ACTIONS_MIN,self.N_ACTIONS_MIN+nActions)],
      self.STATE_TERMINAL_R: [self.ACTION_RIGHT], 
      self.STATE_TERMINAL_L: [self.ACTION_LEFT] }
    # Rewards
    self.defaultReward = 0.0
    self.agentState = self.startState
    
  def step(self, action):
    self.agentState = self.stateTransitions[self.agentState][action]
    if(self.agentState == self.STATE_TERMINAL_L):
      reward = np.random.normal(self.mu, self.std)
    else:
      reward = self.defaultReward
    if(self.agentState in self.terminalStates):
      done = True
    else:
      done = False
    return self.agentState, reward, done
    
  def reset(self):
    self.agentState = self.startState
    return self.agentState
    
  def getAvailableActions(self, state=None):
    if state is None:
      return self.stateActionMapping[self.agentState]
    else:
      return self.stateActionMapping[state]
  
  def printEnv(self):
    pass