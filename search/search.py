# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    nodeStack = util.Stack()
    nodeStack.push((problem.getStartState(), []))

    visitedStates = set()

    while not nodeStack.isEmpty():
        currState, pathToState = nodeStack.pop()

        # If this currState is Goal state return its actions.
        if problem.isGoalState(currState):
            return pathToState

        # Marking the state as visited
        visitedStates.add(currState)

        # Get the successors of the current state
        for adjacentState, move, _ in problem.getSuccessors(currState):
            if adjacentState not in visitedStates:
                # Append the move to reach the adjacent state to the current path list
                newPath = pathToState + [move]
                nodeStack.push((adjacentState, newPath))

    # If the goal state isn't reached, return an empty list
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    stateQueue = util.Queue()
    stateQueue.push((problem.getStartState(), []))

    visitedStates = set()
    visitedStates.add(problem.getStartState())

    while not stateQueue.isEmpty():
        currState, pathToState = stateQueue.pop()

        # If this currState is Goal state, return its actions.
        if problem.isGoalState(currState):
            return pathToState

        # Get the successors of the current state
        for adjacentState, move, _ in problem.getSuccessors(currState):
            if adjacentState not in visitedStates:
                # Append the move to reach the adjacent state to the current path list
                newPath = pathToState + [move]
                stateQueue.push((adjacentState, newPath))
                visitedStates.add(adjacentState)

    # If the goal state isn't reached, return an empty list
    return []



def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    costQueue = util.PriorityQueue()

    # Initialize the queue with the start state and cost 0
    costQueue.push((problem.getStartState(), []), 0)

    visitedStates = set()

    while not costQueue.isEmpty():
        currState, pathToState = costQueue.pop()

        # If this currState is the Goal state, return its actions.
        if problem.isGoalState(currState):
            return pathToState

        if currState not in visitedStates:
            visitedStates.add(currState)
            for adjacentState, move, stepCost in problem.getSuccessors(currState):
                if adjacentState not in visitedStates:
                    newPath = pathToState + [move]
                    # Calculate the total cost for newPath
                    cumulativeCost = problem.getCostOfActions(newPath)
                    costQueue.push((adjacentState, newPath), cumulativeCost)

    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    costQueue = util.PriorityQueue()

    # Start state with cost as heuristic value only because actual cost to start is 0
    initialState = problem.getStartState()
    costQueue.push((initialState, []), heuristic(initialState, problem))

    visitedNodes = set()

    while not costQueue.isEmpty():
        currNode, pathToNode = costQueue.pop()

        # If this currNode is the Goal state, return its actions.
        if problem.isGoalState(currNode):
            return pathToNode

        if currNode not in visitedNodes:
            visitedNodes.add(currNode)
            for adjacentNode, move, stepCost in problem.getSuccessors(currNode):
                if adjacentNode not in visitedNodes:
                    newPath = pathToNode + [move]
                    # Calculate the total cost for newPath which is actual cost + heuristic
                    combinedCost = problem.getCostOfActions(newPath) + heuristic(adjacentNode, problem)
                    costQueue.push((adjacentNode, newPath), combinedCost)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
