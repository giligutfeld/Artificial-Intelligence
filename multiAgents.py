"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Gili Gutfeld
Student ID: 209284512

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math
import gameUtil as game

from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 1 # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):

        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """

        def min_max_value(state, depth):

            # switch the turn
            state.switch_turn(state.turn)

            # if the state is terminal or the depth is 0, return the evaluated value
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state)

            # the turn of the AI (max)
            elif state.turn == game.AI:
                successors = state.getLegalActions(game.AI)
                max_value = -math.inf

                # for each successor call the min_max_value function and check if it has the max value
                for successor in successors:
                    new_state = state.generateSuccessor(game.AI, successor)
                    max_value = max(min_max_value(new_state, depth - 1), max_value)
                return max_value

            # the turn of the player (min)
            else:
                successors = state.getLegalActions(game.PLAYER)
                min_value = math.inf

                # for each successor call the min_max_value function and check if it has the min value
                for successor in successors:
                    new_state = state.generateSuccessor(game.PLAYER, successor)
                    min_value = min(min_max_value(new_state, depth - 1), min_value)
                return min_value

        solution = None
        max_val = -math.inf

        # for each action check if it has the max value
        for action in gameState.getLegalActions(game.AI):
            new_state = gameState.generateSuccessor(game.AI, action)
            val = min_max_value(new_state, self.depth - 1)

            # check if we got better solution
            if val > max_val:
                max_val = val
                solution = action

        return solution


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """

        def max_value(state, a, b, depth):

            # switch the turn
            state.switch_turn(state.turn)

            # if the state is terminal or the depth is 0, return the evaluated value
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state)

            # get the legal actions of the AI
            successors = state.getLegalActions(game.AI)
            max_val = -math.inf

            # for each successor call the min_value function and check if it has the max value
            for successor in successors:
                new_state = state.generateSuccessor(game.AI, successor)
                max_val = max(min_value(new_state, a, b, depth - 1), max_val)

                # pruning
                if max_val > b:
                    return max_val
                a = max(a, max_val)

            return max_val

        def min_value(state, a, b, depth):

            # switch the turn
            state.switch_turn(state.turn)

            # if the state is terminal or the depth is 0, return the evaluated value
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state)

            # get the legal actions of the player
            successors = state.getLegalActions(game.AI)
            min_val = math.inf

            # for each successor call the min_value function and check if it has the max value
            for successor in successors:
                new_state = state.generateSuccessor(game.PLAYER, successor)
                min_val = min(max_value(new_state, a, b, depth - 1), min_val)

                # pruning
                if min_val < a:
                    return min_val
                b = min(b, min_val)

            return min_val

        solution = None
        max_val = -math.inf

        # for each action check if it has the max value
        for action in gameState.getLegalActions(game.AI):
            new_state = gameState.generateSuccessor(game.AI, action)
            val = min_value(new_state, -math.inf, math.inf, self.depth - 1)

            # check if we got better solution
            if val > max_val:
                max_val = val
                solution = action

        return solution

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def max_value(state, depth):

            # switch the turn
            state.switch_turn(state.turn)

            # if the state is terminal or the depth is 0, return the evaluated value
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state)

            # get the legal actions of the AI
            successors = state.getLegalActions(game.AI)
            v = -math.inf

            # for each successor call the exp_value function and check if it has the max value
            for successor in successors:
                new_state = state.generateSuccessor(game.AI, successor)
                v = max(exp_value(new_state, depth - 1), v)

            return v

        def exp_value(state, depth):

            # switch the turn
            state.switch_turn(state.turn)

            # if the state is terminal or the depth is 0, return the evaluated value
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state)

            # get the legal actions of the player
            successors = state.getLegalActions(game.AI)

            # initialize v to 0 and calculate the probability to choose an action
            v = 0
            p = 1 / len(successors)

            # return weighted average of all successors values
            for successor in successors:
                new_state = state.generateSuccessor(game.PLAYER, successor)
                v += p * max_value(new_state, depth - 1)

            return v

        solution = None
        max_val = -math.inf

        # for each action check if it has the max value
        for action in gameState.getLegalActions(game.AI):
            new_state = gameState.generateSuccessor(game.AI, action)
            val = exp_value(new_state, self.depth - 1)

            # check if we got better solution
            if val > max_val:
                max_val = val
                solution = action

        return solution
