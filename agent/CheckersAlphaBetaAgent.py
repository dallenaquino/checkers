from typing import Iterable

from CheckersGame import CheckersGameState, PromotionJumpMove, JumpMove, PushMove, PromotionPushMove, \
    CheckersMove, RED_PIECE, RED_KING, BLACK_PIECE, BLACK_KING, CheckersMoveInfo
from agent.AlphaBetaAgent import AlphaBetaAgent
from util import board_positions


class CheckersAlphaBetaAgent(AlphaBetaAgent):
    def eval(self, state: CheckersGameState):
        if state.is_terminal():
            return 0 if (winner:= state.get_winner()) is None else -1 if winner else 1
        board = state.board
        # dark_squares = tuple((i, j) for i, j in itertools.permutations(range(8), 2) if (i + j) % 2)
        counts = [0] * 4
        for r, c in board_positions:
            piece = board[4 * r + c // 2]
            if piece > BLACK_KING:
                continue
            counts[piece] += 1
        value = 2 * (counts[BLACK_KING] - counts[RED_KING]) + counts[BLACK_PIECE] - counts[RED_PIECE]
        return value / sum(counts)

    def reorder_moves(self, state, moves: Iterable[CheckersMove]) -> Iterable[CheckersMove]:
        def move_priority(move_info: CheckersMoveInfo):
            move = move_info.move
            final_coord = move.compute_final_coord()
            priority = 0
            if isinstance(move, JumpMove):
                priority -= 2
                priority -= len(move.actions)
                if isinstance(move, PromotionJumpMove):
                    priority -= 2
            elif isinstance(move, PushMove):
                if isinstance(move, PromotionPushMove):
                    priority -= 1
            else:
                return 5
            if final_coord[1] in (0, 7):
                priority -= 1
            return priority

        return list(sorted(moves, key=move_priority))