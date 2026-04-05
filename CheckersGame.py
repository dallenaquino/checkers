import copy
import itertools
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Generator, Literal

from GameEngine import GameEngine
from GameState import GameState
from util import board_positions


class Actions(Enum):
    PUSH_NW = auto()
    PUSH_NE = auto()
    PUSH_SW = auto()
    PUSH_SE = auto()
    JUMP_NW = auto()
    JUMP_NE = auto()
    JUMP_SW = auto()
    JUMP_SE = auto()

class CheckerboardCodes(Enum):
    RED_PIECE = 0
    RED_KING = auto()
    BLACK_PIECE = auto()
    BLACK_KING = auto()
    LIGHT_SQUARE = auto()
    DARK_SQUARE = auto()

    def promote(self):
        if self == CheckerboardCodes.RED_PIECE:
            return CheckerboardCodes.RED_KING
        elif self == CheckerboardCodes.BLACK_PIECE:
            return CheckerboardCodes.BLACK_KING
        else:
            return self

    def __str__(self):
        if self == self.RED_PIECE:
            return "RP"
        if self == self.RED_KING:
            return "RK"
        if self == self.BLACK_PIECE:
            return "BP"
        if self == self.BLACK_KING:
            return "BK"
        if self == self.LIGHT_SQUARE:
            return "LS"
        if self == self.DARK_SQUARE:
            return "DS"

    def __repr__(self):
        return self.__str__()

    def is_king(self):
        return self in (self.BLACK_KING, self.RED_KING)


class CheckersMove:
    def __init__(self, x, y):
        self.starting_coords: tuple[int, int] = x, y

    @staticmethod
    def compute_next_coord(starting_pos: tuple[int, int], action: Actions) -> tuple[int, int]:
        if action == Actions.PUSH_NW:
            deltas = (-1, -1)
        elif action == Actions.PUSH_NE:
            deltas = (-1, 1)
        elif action == Actions.PUSH_SW:
            deltas = (1, -1)
        elif action == Actions.PUSH_SE:
            deltas = (1, 1)
        elif action == Actions.JUMP_NW:
            deltas = (-2, -2)
        elif action == Actions.JUMP_NE:
            deltas = (-2, 2)
        elif action == Actions.JUMP_SW:
            deltas = (2, -2)
        elif action == Actions.JUMP_SE:
            deltas = (2, 2)
        else:
            deltas = (0, 0)
        return starting_pos[0] + deltas[0], starting_pos[1] + deltas[1]
    
    @staticmethod
    def action_from_delta(delta_row, delta_col):
        deltas = (delta_row, delta_col)
        if deltas == (-1, -1):
            return Actions.PUSH_NW
        elif deltas == (-1, 1):
            return Actions.PUSH_NE
        elif deltas == (1, -1):
            return Actions.PUSH_SW
        elif deltas == (1, 1):
            return Actions.PUSH_SE
        elif deltas == (-2, -2):
            return Actions.JUMP_NW
        elif deltas == (-2, 2):
            return Actions.JUMP_NE
        elif deltas == (2, -2):
            return Actions.JUMP_SW
        elif deltas == (2, 2):
            return Actions.JUMP_SE
        else:
            return None

    @abstractmethod
    def compute_final_coord(self) -> tuple[int, int]:
        pass


class PushMove(CheckersMove):
    def __init__(self, start_x, start_y, action: Actions):
        super().__init__(start_x, start_y)
        self.action = action

    def compute_final_coord(self) -> tuple[int, int]:
        return self.compute_next_coord(self.starting_coords, self.action)

class JumpMove(CheckersMove):
    def __init__(self, x, y, *args):
        super().__init__(x, y)
        self.actions = [*args]

    def compute_final_coord(self) -> tuple[int, int]:
        pos = self.starting_coords
        for action in self.actions:
            pos = self.compute_next_coord(pos, action)
        return pos

class PromotionMove(ABC):
    pass

class PromotionPushMove(PushMove, PromotionMove):
    pass

class PromotionJumpMove(JumpMove, PromotionMove):
    pass

class CheckersGameState(GameState):
    def __init__(self, empty=False):
        super().__init__()
        self.board = [[CheckerboardCodes.DARK_SQUARE if (i + j) % 2 else CheckerboardCodes.LIGHT_SQUARE for j in range(8)] for i in range(8)]
        if empty:
            return
        for i in range(3):
            for j in range((i + 1) % 2, 8, 2):
                self.board[i][j] = CheckerboardCodes.BLACK_PIECE
                self.board[-i - 1][-j - 1] = CheckerboardCodes.RED_PIECE

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self.current_player_index == other.current_player_index and self.board == other.board

    def get_legal_moves(self):
        return [*self._get_legal_moves_helper()]

    def is_terminal(self):
        return next(self._get_legal_moves_helper(), None) is None

    def get_winner(self):
        if not self.is_terminal():
            return None
        # No available moves: current player lost
        return (self.current_player_index + 1) % 2

    def __deepcopy__(self, memodict={}):
        new_state = CheckersGameState()
        new_state.current_player_index = self.current_player_index
        new_state.board = [row[:] for row in self.board]
        return new_state

    def generate_successor(self, move: CheckersMove) -> GameState:
        r, c = move.starting_coords
        expected_pieces = []
        if self.current_player_index == 0:
            expected_pieces.append(CheckerboardCodes.BLACK_PIECE)
            if not isinstance(move, PromotionMove):
                expected_pieces.append(CheckerboardCodes.BLACK_KING)
        else:
            expected_pieces.append(CheckerboardCodes.RED_PIECE)
            if not isinstance(move, PromotionMove):
                expected_pieces.append(CheckerboardCodes.RED_KING)
        if self.board[r][c] not in expected_pieces:
            raise Exception("Square does not contain expected piece type/color")
        if isinstance(move, PushMove):
            end_r, end_c = move.compute_next_coord((r, c), move.action)
            new_state = copy.deepcopy(self)
            prev_piece = new_state.board[r][c]
            if isinstance(move, PromotionMove):
                new_state.board[end_r][end_c] = prev_piece.promote()
            else:
                new_state.board[end_r][end_c] = prev_piece
            new_state.board[r][c] = CheckerboardCodes.DARK_SQUARE
        elif isinstance(move, JumpMove):
            new_state = copy.deepcopy(self)
            prev_piece = new_state.board[r][c]
            prev_r, prev_c = (r, c)
            if len(move.actions) == 0:
                raise RuntimeError("No more possible moves. Game should have ended")
            for action in move.actions:
                new_r, new_c = move.compute_next_coord((prev_r, prev_c), action)
                mid_r, mid_c = ((x + y) // 2 for x, y in zip((prev_r, prev_c), (new_r, new_c)))
                new_state.board[new_r][new_c] = prev_piece
                new_state.board[prev_r][prev_c] = CheckerboardCodes.DARK_SQUARE
                new_state.board[mid_r][mid_c] = CheckerboardCodes.DARK_SQUARE
                prev_r, prev_c = new_r, new_c
            if isinstance(move, PromotionMove):
                new_state.board[new_r][new_c] = prev_piece.promote()

        else:
            raise TypeError(f"Unknown move type: {type(move)}")
        new_state.current_player_index = (self.current_player_index + 1) % 2
        return new_state


    def _get_legal_moves_helper(self, start_pos=None) -> Generator[CheckersMove, None, None]:
        is_black = lambda c: c in (CheckerboardCodes.BLACK_KING, CheckerboardCodes.BLACK_PIECE)
        is_red = lambda c: c in (CheckerboardCodes.RED_KING, CheckerboardCodes.RED_PIECE)
        is_black_at = lambda x: is_black(self.board[x[0]][x[1]])
        is_red_at = lambda x: is_red(self.board[x[0]][x[1]])
        black_positions = list(filter(is_black_at, itertools.permutations(range(8), 2)))
        red_positions = list(filter(is_red_at, itertools.permutations(range(8), 2)))
        if self.current_player_index == 0:
            default_delta_i = 1
            curr_player_positions = black_positions
            is_opponent = is_red
            king_row = 7
        else:
            # Regular pieces move up
            default_delta_i = -1
            curr_player_positions = red_positions
            is_opponent = is_black
            king_row = 0
        if start_pos is not None:
            curr_player_positions = [start_pos]
        found_jumps = False
        push_moves = []
        for i, j in curr_player_positions:
            current_piece = self.board[i][j]
            deltas_i = (-1, 1) if current_piece.is_king() else (default_delta_i,)
            for delta_i in deltas_i:
                new_i = i + delta_i
                if not (0 <= new_i < 8):
                    continue
                # Must jump if possible. Look for jump moves first
                for delta_j in -1, 1:
                    new_j = j + delta_j
                    if not (0 <= new_j < 8):
                        continue
                    if is_opponent(self.board[new_i][new_j]):
                        jump_i = i + 2 * delta_i
                        jump_j = j + 2 * delta_j
                        if not ((0 <= jump_i < 8) and (0 <= jump_j < 8)):
                            continue
                        if self.board[jump_i][jump_j] != CheckerboardCodes.DARK_SQUARE:
                            continue
                        found_jumps = True
                        action = CheckersMove.action_from_delta(2 * delta_i, 2 * delta_j)
                        # if piece reaches far side of the board and is not already a king,
                        # a promotion will happen and the turn will end, so no need to look for additional jumps
                        if jump_i == king_row and not self.board[i][j].is_king():
                            yield PromotionJumpMove(i, j, action)
                            continue
                        new_state = copy.deepcopy(self)
                        new_state.board[i][j] = CheckerboardCodes.DARK_SQUARE
                        new_state.board[new_i][new_j] = CheckerboardCodes.DARK_SQUARE
                        new_state.board[jump_i][jump_j] = self.board[i][j]

                        additional_moves = new_state._get_legal_moves_helper(start_pos=(jump_i, jump_j))
                        jump_moves = [*filter(lambda m: isinstance(m, JumpMove), additional_moves)]
                        if len(jump_moves):
                            for m in jump_moves:
                                if isinstance(m, PromotionJumpMove):
                                    yield PromotionJumpMove(i, j, action, *m.actions)
                                else:
                                    yield JumpMove(i, j, action, *m.actions)
                        else:
                            yield JumpMove(i, j, action)
                # Push moves only allowed if no jumps are possible
                if found_jumps:
                    continue
                # Now look for push moves
                for delta_j in -1, 1:
                    new_j = j + delta_j
                    if not (0 <= new_j < 8):
                        continue
                    if self.board[new_i][new_j] == CheckerboardCodes.DARK_SQUARE:
                        action = CheckersMove.action_from_delta(delta_i, delta_j)
                        move = PromotionPushMove(i, j, action) if new_i == king_row and not self.board[i][j].is_king() else PushMove(i, j, action)
                        push_moves.append(move)
        if not found_jumps:
            yield from push_moves

class StateLogger:
    def __init__(self, starting_player=0):
        self._starting_player = starting_player
        self.score = 0
        self.past_states = []

    @property
    def starting_player(self):
        return self._starting_player

    @starting_player.setter
    def starting_player(self, val):
        self._starting_player = int(val)

    def log(self, state: CheckersGameState, score=None):
        self.past_states.append(self.tokenize(state))
        if score is not None:
            self.score = score

    @staticmethod
    def tokenize(state: CheckersGameState):
        token = 0
        board = state.board
        for i, j in board_positions():
            piece = board[i][j]
            token <<= 4
            if piece.value <= CheckerboardCodes.BLACK_KING.value:
                token |= (1 << piece.value)

        return token

    def save(self, file_name, endianness: Literal['big'] | Literal['little'] = 'big'):
        with open(file_name, 'ab') as f:
            f.write(len(self.past_states).to_bytes(2, byteorder=endianness))
            f.write(int.to_bytes(self.starting_player, byteorder=endianness))
            for vector in self.past_states:
                f.write(vector.to_bytes(16, byteorder=endianness))
            f.write(self.score.to_bytes(byteorder=endianness, signed=True))

class CheckersGame(GameEngine):
    def __init__(self, *agents):
        super().__init__(*agents)
        self.current_state = CheckersGameState()

    def make_move(self, move: CheckersMove):
        self.current_state = self.current_state.generate_successor(move)
