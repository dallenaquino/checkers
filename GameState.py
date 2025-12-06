from abc import ABC, abstractmethod


class GameState(ABC):
    def __init__(self):
        self.current_player_index = 0

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_legal_moves(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def generate_successor(self, move):
        pass

    @abstractmethod
    def get_winner(self):
        pass
