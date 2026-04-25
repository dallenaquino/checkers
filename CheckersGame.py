from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from enum import Enum, auto, IntEnum
from typing import Generator, Literal, Self, Iterable, Optional, Sized

from GameEngine import GameEngine
from GameState import GameState, MoveInfo
from util import board_positions


class CheckersMoveInfo(MoveInfo):
    def __init__(self, move: CheckersMove, result: Optional[CheckersGameState]=None):
        super().__init__(move, result)


class Actions(Enum):
    PUSH_NW = auto()
    PUSH_NE = auto()
    PUSH_SW = auto()
    PUSH_SE = auto()
    JUMP_NW = auto()
    JUMP_NE = auto()
    JUMP_SW = auto()
    JUMP_SE = auto()

class CheckerboardCodes(IntEnum):
    RED_PIECE = 0
    RED_KING = auto()
    BLACK_PIECE = auto()
    BLACK_KING = auto()
    LIGHT_SQUARE = auto()
    DARK_SQUARE = auto()

    @classmethod
    def promote(cls, value):
        if value == RED_PIECE:
            return RED_KING.value
        elif value == BLACK_PIECE:
            return BLACK_KING.value
        else:
            return value

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

    @classmethod
    def is_king(cls, value):
        return value in (cls.BLACK_KING, cls.RED_KING)


RED_PIECE, RED_KING, BLACK_PIECE, BLACK_KING, LIGHT_SQUARE, DARK_SQUARE = CheckerboardCodes


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

    class CheckersBoardState(bytearray):
        def __init__(self, default_board=True):
            super().__init__(32)
            if not default_board:
                return
            for i, j in board_positions:
                self[4 * i + j // 2] = BLACK_PIECE.value if i < 3 else\
                    RED_PIECE.value if i > 4 else DARK_SQUARE.value

    def __init__(self, moves_memo, construct_board=True):
        super().__init__()
        self.moves_since_capture = 0
        self.board = self.CheckersBoardState(construct_board)
        self.saved_moves = moves_memo

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self.current_player_index == other.current_player_index and self.board == other.board

    def get_legal_moves(self) -> tuple[CheckersMoveInfo]:
        # return tuple(self._get_legal_moves_helper())
        key = self._get_key()
        if key not in self.saved_moves:
            self.saved_moves[key] = tuple(self._get_legal_moves_helper())
        return self.saved_moves[key]

    def is_terminal(self):
        return self.moves_since_capture > 40 or len(self.get_legal_moves()) == 0

    def get_winner(self):
        if not self.is_terminal():
            return None
        if self.moves_since_capture > 40:
            return None
        # No available moves: current player lost
        return (self.current_player_index + 1) % 2

    def __deepcopy__(self, memodict={}):
        new_state = CheckersGameState(self.saved_moves, construct_board=False)
        new_state.current_player_index = self.current_player_index
        new_state.board = self.board[:]
        return new_state

    def generate_successor(self, move_info: CheckersMoveInfo) -> CheckersGameState:
        if move_info.result is not None:
            return move_info.result
        move = move_info.move
        r, c = move.starting_coords
        expected_pieces = []
        if self.current_player_index == 0:
            expected_pieces.append(BLACK_PIECE)
            if not isinstance(move, PromotionMove):
                expected_pieces.append(BLACK_KING)
        else:
            expected_pieces.append(RED_PIECE)
            if not isinstance(move, PromotionMove):
                expected_pieces.append(RED_KING)
        flat_idx = 4 * r + c // 2
        if self.board[flat_idx] not in expected_pieces:
            raise Exception("Square does not contain expected piece type/color")
        new_state = copy.deepcopy(self)
        if isinstance(move, PushMove):
            new_state.moves_since_capture = self.moves_since_capture + 1
            end_r, end_c = move.compute_next_coord((r, c), move.action)
            prev_piece = new_state.board[flat_idx]
            end_flat_idx = 4 * end_r + end_c // 2
            new_state.board[end_flat_idx] = CheckerboardCodes.promote(prev_piece) if isinstance(move, PromotionMove) else prev_piece
            new_state.board[flat_idx] = DARK_SQUARE
        elif isinstance(move, JumpMove):
            new_state.moves_since_capture = 0
            prev_piece = new_state.board[flat_idx]
            prev_r, prev_c = (r, c)
            if len(move.actions) == 0:
                raise RuntimeError("No more possible moves. Game should have ended")
            for action in move.actions:
                new_r, new_c = move.compute_next_coord((prev_r, prev_c), action)
                mid_r, mid_c = ((x + y) // 2 for x, y in zip((prev_r, prev_c), (new_r, new_c)))
                new_flat_idx = 4 * new_r + new_c // 2
                new_state.board[new_flat_idx] = prev_piece
                new_state.board[4 * prev_r +  prev_c // 2] = DARK_SQUARE
                new_state.board[4 * mid_r + mid_c // 2] = DARK_SQUARE
                prev_r, prev_c = new_r, new_c
            if isinstance(move, PromotionMove):
                new_state.board[new_flat_idx] = CheckerboardCodes.promote(prev_piece)

        else:
            raise TypeError(f"Unknown move type: {type(move)}")
        new_state.current_player_index = (self.current_player_index + 1) % 2
        move_info.result = new_state
        return new_state


    def _get_legal_moves_helper(self, start_pos=None) -> Generator[CheckersMoveInfo, None, None]:
        is_black = lambda c: c in (BLACK_KING, BLACK_PIECE)
        is_red = lambda c: c in (RED_KING, RED_PIECE)
        is_black_at = lambda x: is_black(self.board[4 * x[0] + x[1] // 2])
        is_red_at = lambda x: is_red(self.board[4 * x[0] + x[1] // 2])
        black_positions = filter(is_black_at, board_positions)
        red_positions = filter(is_red_at, board_positions)
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
            current_piece = self.board[4 * i + j // 2]
            deltas_i = (-1, 1) if CheckerboardCodes.is_king(current_piece) else (default_delta_i,)
            for delta_i in deltas_i:
                new_i = i + delta_i
                if not (0 <= new_i < 8):
                    continue
                # Must jump if possible. Look for jump moves first
                for delta_j in -1, 1:
                    new_j = j + delta_j
                    if not (0 <= new_j < 8):
                        continue
                    if is_opponent(self.board[4 * new_i + new_j // 2]):
                        jump_i = i + 2 * delta_i
                        jump_j = j + 2 * delta_j
                        if not ((0 <= jump_i < 8) and (0 <= jump_j < 8)):
                            continue
                        if self.board[4 * jump_i + jump_j // 2] != DARK_SQUARE:
                            continue
                        found_jumps = True
                        action = CheckersMove.action_from_delta(2 * delta_i, 2 * delta_j)
                        move_cls = PromotionJumpMove if jump_i == king_row and not CheckerboardCodes.is_king(self.board[4 * i + j // 2]) else JumpMove
                        move = move_cls(i, j, action)
                        move_info = CheckersMoveInfo(move)
                        new_state = self.generate_successor(move_info)
                        move_info.result = new_state

                        # if piece reaches far side of the board and is not already a king,
                        # a promotion will happen and the turn will end, so no need to look for additional jumps
                        if isinstance(move, PromotionMove):
                            yield move_info
                            continue

                        # Set new_state player index to current one for recursive call
                        next_player = new_state.current_player_index
                        new_state.current_player_index = self.current_player_index
                        additional_moves_gen = new_state._get_legal_moves_helper(start_pos=(jump_i, jump_j))
                        first_move_info = next(additional_moves_gen, None)
                        if first_move_info is None or not isinstance(first_move_info.move, JumpMove):
                            new_state.current_player_index = next_player
                            yield move_info
                            continue
                        additional_moves = [first_move_info]
                        additional_moves.extend(additional_moves_gen)
                        new_state.current_player_index = next_player
                        for move_info in additional_moves:
                            m = move_info.move
                            s = move_info.result
                            cls = type(m)
                            yield MoveInfo(cls(i, j, action, *m.actions), s)
                # Push moves only allowed if no jumps are possible
                if found_jumps:
                    continue
                # Now look for push moves
                for delta_j in -1, 1:
                    new_j = j + delta_j
                    if not (0 <= new_j < 8):
                        continue
                    neighbor_piece = self.board[4 * new_i + new_j // 2]
                    if neighbor_piece == DARK_SQUARE:
                        action = CheckersMove.action_from_delta(delta_i, delta_j)
                        move = PromotionPushMove(i, j, action) if new_i == king_row and not CheckerboardCodes.is_king(self.board[4 * i + j // 2]) else PushMove(i, j, action)
                        push_moves.append(CheckersMoveInfo(move))
        if not found_jumps:
            yield from push_moves

    def _get_key(self):
        return bytes(self.board), self.current_player_index, self.moves_since_capture


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
        board = state.board
        compressed_states = bytearray(16)
        for i in range(0, len(board), 2):
            token = 0
            for j in range(2):
                token <<= 4
                piece = board[i + j]
                if piece <= BLACK_KING:
                    token |= (1 << piece)
            compressed_states[i // 2] = token

        return bytes(compressed_states), state.moves_since_capture

    def save(self, file_name, endianness: Literal['big'] | Literal['little'] = 'big'):
        with open(file_name, 'ab') as f:
            f.write(len(self.past_states).to_bytes(2, byteorder=endianness))
            f.write(int.to_bytes(self.starting_player, byteorder=endianness))
            for vector, msc in self.past_states:
                f.write(vector)
                f.write(int.to_bytes(msc, byteorder=endianness))
            f.write(self.score.to_bytes(byteorder=endianness, signed=True))

class CheckersGame(GameEngine):
    def __init__(self, *agents):
        super().__init__(*agents)
        self.move_memo = {}
        self.current_state = CheckersGameState(self.move_memo)

    def make_move(self, move: CheckersMoveInfo):
        self.current_state = self.current_state.generate_successor(move)

    def get_next_move(self) -> CheckersMoveInfo:
        return super().get_next_move()
