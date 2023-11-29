# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Observations to be noted:
        # 1. Pacman needs to eat the nearest food.
        # 2. Go away from the ghost.
        # 3. Go to a scared ghost to eat it.
        # 4. Shouldn't stop.
        # This way we can maximize the score.


        # Move towards the closest food
        food_reward = 0
        min_food_distance = float('inf')
        food_list = newFood.asList()
        if food_list:
            all_food_distances = [manhattanDistance(newPos, food) for food in food_list]
            min_food_distance = min(all_food_distances)
        food_reward += 1 / (min_food_distance + 1)

        # Avoid close ghosts and chase scared ghosts
        ghost_penalty = 0
        min_ghost_distance = float('inf')
        for ghostState in newGhostStates:
            min_ghost_distance = min(min_ghost_distance, manhattanDistance(newPos, ghostState.getPosition()))

        # scaling the ghost_penalty to twice so that it prioritize the ghost
        if min(newScaredTimes) > 0:
            ghost_penalty = ghost_penalty + 2 / (min_ghost_distance + 1)
        else:
            ghost_penalty = ghost_penalty - 2 / (min_ghost_distance + 1)

        # Penalize for stopping (return -infinity as stopping shouldn't be an option as a state)
        if action == Directions.STOP:
            return -float('inf')

        # Combine rewards and penalties for the final score
        return successorGameState.getScore() + food_reward + ghost_penalty

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Observations:
        # 1. maximizes a pacman outcome thinking ghost play optimally.
        # 2. max_value: best move for the pacman.
        # 3. min_value: best move for the ghost.
        def MINIMAX_DECISION(state):
            actions = state.getLegalActions(0)
            best_value = float("-inf")
            best_action = Directions.STOP

            for action in actions:
                value = MIN_VALUE(state.generateSuccessor(0, action), 1, self.depth)
                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action

        def MAX_VALUE(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float("-inf")
            for a in state.getLegalActions(0):
                v = max(v, MIN_VALUE(state.generateSuccessor(0, a), 1, depth))
            return v

        def MIN_VALUE(state, agent_index, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float("inf")
            for a in state.getLegalActions(agent_index):
                # checking if this is the last agent
                if agent_index == state.getNumAgents() - 1:
                    v = min(v, MAX_VALUE(state.generateSuccessor(agent_index, a), depth - 1))
                else:
                    v = min(v, MIN_VALUE(state.generateSuccessor(agent_index, a), agent_index + 1, depth))
            return v

        return MINIMAX_DECISION(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Observations:
        # 1. After each action evaluation, the alpha and beta values are updated
        # 2. Pruned when current values cross predetermined alpha or beta bounds


        def ALPHA_BETA_SEARCH(state):
            alpha = float("-inf")
            beta = float("inf")
            best_value = float("-inf")
            best_action = Directions.STOP

            for action in state.getLegalActions(0):
                value = max(best_value, MIN_VALUE(state.generateSuccessor(0, action), 1, self.depth, alpha, beta))
                if value > best_value:
                    best_value = value
                    best_action = action

                alpha = max(alpha, value)  # Update alpha with the best value found so far
                if value > beta:  # Prune the branch
                    return value
            return best_action

        def MAX_VALUE(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float("-inf")
            for a in state.getLegalActions(0):
                v = max(v, MIN_VALUE(state.generateSuccessor(0, a), 1, depth, alpha, beta))
                if v > beta:  # Prune the branch
                    return v
                alpha = max(alpha, v)  # Update alpha with the best value found so far

            return v

        def MIN_VALUE(state, agent_index, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float("inf")
            for a in state.getLegalActions(agent_index):
                if agent_index == state.getNumAgents() - 1:
                    v = min(v, MAX_VALUE(state.generateSuccessor(agent_index, a), depth - 1, alpha, beta))
                else:
                    v = min(v, MIN_VALUE(state.generateSuccessor(agent_index, a), agent_index + 1, depth, alpha, beta))
                if v < alpha:  # Prune the branch
                    return v
                beta = min(beta, v)  # Update beta with the best value found so far

            return v

        return ALPHA_BETA_SEARCH(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Observations:
        # 1. Here we have pacman maximizing strategy and the ghosts' probabilistic behavior.
        # 2. Ghost actions are evaluated based on a uniform probability distribution.
        # 3. Optimal move is determined by the action leading to the highest estimated value.

        def value(state, agent_index, depth):
            # Base case
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Pacman
            if agent_index == 0:
                return max_value(state, depth)
            # Ghosts
            else:
                return exp_value(state, agent_index, depth)

        def max_value(state, depth):
            v = float("-inf")
            for action in state.getLegalActions(0):
                v = max(v, value(state.generateSuccessor(0, action), 1, depth))
            return v

        def exp_value(state, agent_index, depth):
            v = 0
            legal_actions = state.getLegalActions(agent_index)
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                if agent_index == gameState.getNumAgents() - 1:
                    next_depth = depth - 1
                else:
                    next_depth = depth
                v += prob * value(state.generateSuccessor(agent_index, action),
                                  (agent_index + 1) % gameState.getNumAgents(), next_depth)
            return v

        best_value = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions():
            new_value = value(gameState.generateSuccessor(0, action), 1, self.depth)
            if new_value > best_value:
                best_value = new_value
                best_action = action

        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Observations to be noted:
    # 1. Pacman needs to eat the nearest food.
    # 2. Go away from the ghost.
    # 3. Go to a scared ghost to eat it.
    # 4. Rewarded when completing the pallets.
    # This way we can maximize the score.

    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    num_of_capsules = len(currentGameState.getCapsules())

    food_reward = 0
    min_food_distance = float('inf')
    food_list = food.asList()
    if food_list:
        all_food_distances = [manhattanDistance(pos, food) for food in food_list]
        min_food_distance = min(all_food_distances)
    food_reward += 1 / (min_food_distance + 1)

    # Avoid close ghosts and chase scared ghosts
    ghost_penalty = 0
    min_ghost_distance = float('inf')
    for ghostState in ghost_states:
        min_ghost_distance = min(min_ghost_distance, manhattanDistance(pos, ghostState.getPosition()))

    # scaling the ghost_penalty to twice so that it prioritize the ghost
    if min(scared_times) > 0:
        ghost_penalty = ghost_penalty + 2 / (min_ghost_distance + 1)
    else:
        ghost_penalty = ghost_penalty - 2 / (min_ghost_distance + 1)

    capsule_reward = 0
    # print(num_of_capsules)
    capsule_reward = capsule_reward + 2 / (num_of_capsules + 1)

    # Combine rewards and penalties for the final score
    return currentGameState.getScore() + food_reward + ghost_penalty + capsule_reward



# Abbreviation
better = betterEvaluationFunction
