# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        # My intuition:
        # we will here iterate through the given number of iterations and calculate the max Q values for every state.
        # We will use these values afterward for every state

        for iteration in range(self.iterations):
            storeValues = util.Counter()

            for state in self.mdp.getStates():
                values = []
                ilegalAction = self.mdp.isTerminal(state)
                if not ilegalAction:
                    for action in self.mdp.getPossibleActions(state):
                        values.append(self.computeQValueFromValues(state, action))
                    if len(values) != 0:
                        storeValues[state] = max(values)
                    else:
                        storeValues[state] = 0.0
            self.values = storeValues




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def bellman(self, state, action, qValueSummation, transitionState, transitionProbability, valueOfNextState):
        rewardGiven = self.mdp.getReward(state, action, transitionState)
        discountGiven = self.discount
        summation = qValueSummation + (transitionProbability * (rewardGiven + (discountGiven * valueOfNextState)))
        return summation
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # My intuition:
        # We use bellman equation here
        # With that equation we will calculate the Q value with the immediate reward and discount factor.

        qValueSummation = 0

        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for transitionState, transitionProbability in transitionStatesAndProbs:
            valueOfNextState = self.getValue(transitionState)
            qValueSummation = self.bellman(state, action, qValueSummation, transitionState, transitionProbability, valueOfNextState)
        return qValueSummation


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # My intuition:
        # We need to return the best action which will have the highest q value for that state

        ilegalAction = self.mdp.isTerminal(state)
        if ilegalAction:
            return None

        actionQs = {}

        for action in self.mdp.getPossibleActions(state):
            actionQs[action] = self.computeQValueFromValues(state, action)

        maxAction = None
        maxQValue = None
        for action, qValue in actionQs.items():
            if maxQValue is None or qValue > maxQValue:
                maxQValue = qValue
                maxAction = action
        return maxAction



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
