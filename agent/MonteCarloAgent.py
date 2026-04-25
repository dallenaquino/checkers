import math
import random

from GameState import GameState, MoveInfo
from agent.Agent import DecisionTreeNode, Agent


class MonteCarloNode(DecisionTreeNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wins = 0
        self.simulations = 0


class MonteCarloAgent(Agent):
    def __init__(self, iterations_per_turn=500):
        self.tree = MonteCarloNode()
        self.select_heuristic = self.ucb1
        self.iterations = iterations_per_turn


    def choose_move(self, game_state: GameState) -> MoveInfo:
        if self.tree.game_state != game_state:
            node = next(filter(lambda c: c.game_state == game_state, self.tree.children), None)
            if node is not None:
                self.tree = node
            else:
                self.tree = MonteCarloNode()
                self.tree.game_state = game_state
                self.generate_children(self.tree)
        for _ in range(self.iterations):
            self.iterate()
        pass
        best = max(self.tree.children, key=lambda n: n.simulations)
        idx = self.tree.children.index(best)
        move = self.tree.moves[idx]
        self.tree = best
        return move

    def iterate(self):
        # Select
        node = self.tree
        while len(node.children) > 0:
            node = max(node.children, key=self.select_heuristic)

        # Expand
        if node.simulations > 0:
            self.generate_children(node)
            if len(node.children):
                node = node.children[0]
        # Simulate
        winner = self.simulate(node)
        # Back-propagate
        root_player = self.tree.game_state.current_player_index

        while node is not None:
            node.simulations += 1
            if root_player == winner:
                node.wins += 1
            node = node.parent

    @staticmethod
    def ucb1(node, c=math.sqrt(2)) -> float:
        if node.simulations == 0:
            return math.inf
        exploitation = node.wins / node.simulations
        exploration = math.sqrt(math.log(node.parent.simulations, math.e) / node.simulations)
        return exploitation + c * exploration

    @staticmethod
    def simulate(start_node: MonteCarloNode):
        state = start_node.game_state
        while not state.is_terminal():
            moves = state.get_legal_moves()
            move = random.choice(moves)
            state = state.generate_successor(move)
        return state.get_winner()


    @staticmethod
    def generate_children(node):
        node.children.clear()
        node.moves = ()
        if node.game_state is None:
            return
        node.moves = node.game_state.get_legal_moves()
        for move_info in node.moves:
            state = node.game_state.generate_successor(move_info)
            new_node = MonteCarloNode(node)
            new_node.game_state = state
            node.children.append(new_node)