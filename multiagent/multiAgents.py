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
        def get_min_food_distance(newPos, newFood):
            foodList = newFood.asList()
            if foodList:
                allFoodDistances = [manhattanDistance(newPos, food) for food in foodList]
                return min(allFoodDistances)
            return 0

        def get_min_ghost_distance(newPos, newGhostStates):
            allGhostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
            return min(allGhostDistances)

        def is_ghost_scared(newScaredTimes):
            return min(newScaredTimes) > 0

        def get_evaluation_components(isGhostScared):
            w1, w2, w3 = 1.0, 2.0, 0.0
            if isGhostScared:
                w1, w2, w3 = 1.0, 0.0, 2.0  # If ghosts are scared, prioritize eating them over avoiding them
            return w1, w2, w3

        minFoodDistance = get_min_food_distance(newPos, newFood)
        minGhostDistance = get_min_ghost_distance(newPos, newGhostStates)

        if action == Directions.STOP:
            return -float('inf')  # Discourage STOP action

        isGhostScared = is_ghost_scared(newScaredTimes)
        w1, w2, w3 = get_evaluation_components(isGhostScared)

        # Combine the components to form the evaluation function
        score = successorGameState.getScore()
        evalFunction = (
                score +
                w1 * (1.0 / (minFoodDistance + 1)) -
                w2 * (1.0 / (minGhostDistance + 1)) +
                w3 * (1.0 / (minGhostDistance + 1))
        )
        return evalFunction

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

        def is_terminal_state(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def get_scores(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            scores = []
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                nextAgent, nextDepth = get_next_agent_and_depth(agentIndex, depth, gameState)
                score, _ = minimax(nextAgent, nextDepth, successorGameState)
                scores.append((score, action))
            return scores

        def get_next_agent_and_depth(agentIndex, depth, gameState):
            if agentIndex == gameState.getNumAgents() - 1:
                return 0, depth + 1
            else:
                return agentIndex + 1, depth

        def minimax(agentIndex, depth, gameState):
            if is_terminal_state(gameState, depth):
                return self.evaluationFunction(gameState), None

            scores = get_scores(agentIndex, depth, gameState)

            if not scores:
                return self.evaluationFunction(gameState), None

            return max(scores) if agentIndex == 0 else min(scores)

        _, action = minimax(0, 0, gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def is_terminal_state(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def get_successor(gameState, agentIndex, action):
            return gameState.generateSuccessor(agentIndex, action)

        def get_max_value(gameState, depth, alpha, beta):
            if is_terminal_state(gameState, depth):
                return self.evaluationFunction(gameState), None

            v = float('-inf')
            best_action = None
            for action in gameState.getLegalActions(0):  # 0 is always Pacman
                successor = get_successor(gameState, 0, action)
                score, _ = get_min_value(successor, depth, 1, alpha, beta)
                if score > v:
                    v, best_action = score, action
                alpha = max(alpha, v)
                if v > beta:
                    break  # Prune
            return v, best_action

        def get_min_value(gameState, depth, agentIndex, alpha, beta):
            if is_terminal_state(gameState, depth):
                return self.evaluationFunction(gameState), None

            v = float('inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successor = get_successor(gameState, agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:  # If last ghost
                    score, _ = get_max_value(successor, depth + 1, alpha, beta)
                else:  # If not last ghost
                    score, _ = get_min_value(successor, depth, agentIndex + 1, alpha, beta)
                if score < v:
                    v, best_action = score, action
                beta = min(beta, v)
                if v < alpha:
                    break  # Prune
            return v, best_action

        _, action = get_max_value(gameState, 0, float('-inf'), float('inf'))
        return action

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

        def is_terminal_state(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def get_max_value(gameState, depth):
            if is_terminal_state(gameState, depth):
                return self.evaluationFunction(gameState), None

            v = float('-inf')
            best_action = None
            for action in gameState.getLegalActions(0):  # 0 is always Pacman
                successor = gameState.generateSuccessor(0, action)
                score, _ = get_exp_value(successor, depth, 1)
                if score > v:
                    v, best_action = score, action
            return v, best_action

        def get_exp_value(gameState, depth, agentIndex):
            if is_terminal_state(gameState, depth):
                return self.evaluationFunction(gameState), None

            v = 0
            best_action = None
            num_actions = len(gameState.getLegalActions(agentIndex))
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:  # If last ghost
                    score, _ = get_max_value(successor, depth + 1)
                else:  # If not last ghost
                    score, _ = get_exp_value(successor, depth, agentIndex + 1)
                v += score / num_actions
            return v, best_action

        _, action = get_max_value(gameState, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def getClosestDistance(pos, positions):
        distances = [util.manhattanDistance(pos, p) for p in positions]
        return min(distances) if distances else 0

    pos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Distances to nearest food and ghost
    closestFoodDist = getClosestDistance(pos, foodPos)
    ghostPos = [ghostState.getPosition() for ghostState in ghostStates]
    closestGhostDist = getClosestDistance(pos, ghostPos)

    # Check if ghosts are scared or not
    nonScaredGhosts, scaredGhosts = [], []
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            scaredGhosts.append(ghostState.getPosition())
        else:
            nonScaredGhosts.append(ghostState.getPosition())

    closestScaredGhostDist = getClosestDistance(pos, scaredGhosts)
    closestNonScaredGhostDist = getClosestDistance(pos, nonScaredGhosts)

    # Weights configuration
    w_food = -4.0  # Encourage getting food
    w_ghost = 2.0  # Avoid ghosts
    w_scared_ghost = -5.0  # Encourage chasing scared ghosts
    w_remaining_food = 4.0  # Encourage finishing fast

    # Evaluation formula
    score = currentGameState.getScore()
    evalScore = (
            score +
            w_food * (1.0 / (closestFoodDist + 1)) +
            w_ghost * (1.0 / (closestNonScaredGhostDist + 1)) +
            w_scared_ghost * (1.0 / (closestScaredGhostDist + 1)) +
            w_remaining_food * (1.0 / (len(foodPos) + 1))
    )

    return evalScore

# Abbreviation
better = betterEvaluationFunction
