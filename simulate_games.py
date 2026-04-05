import argparse
import shlex
from argparse import ArgumentParser, FileType

import numpy as np

from CheckersGame import CheckersGame, StateLogger, CheckersGameState, CheckerboardCodes
from agent.Agent import RandomAgent, DeterministicAgent
from agent.CheckersAlphaBetaAgent import CheckersAlphaBetaAgent
from agent.MonteCarloAgent import MonteCarloAgent
from util import board_positions


def agent_type(s: str):
    match s:
        case "random":
            return RandomAgent()
        case "alpha-beta":
            return CheckersAlphaBetaAgent()
        case "monte-carlo":
            return MonteCarloAgent()
        case "deterministic":
            return DeterministicAgent()
        case _:
            raise argparse.ArgumentTypeError(f"Unrecognized agent type: {s}")


def reconstruct_state(vector: [int | bytes], player_index):
    state = CheckersGameState(True)
    if isinstance(vector, int):
        vector = vector.to_bytes(16, byteorder='big')
    for i, (r, c) in enumerate(board_positions()):
        nibble = (vector[i // 2] >> (4 * ((i + 1) % 2))) & 0xf
        if not nibble:
            continue
        offset = -1
        while nibble:
            offset += 1
            nibble >>= 1
        state.board[r][c] = CheckerboardCodes(offset)
    state.current_player_index = player_index
    return state



def sample_states(game_states: list[int], num_states=2):
    n = len(game_states)
    sigma = 0.15
    p = np.arange(n) / (n - 1)
    w = np.exp(-0.5 * ((p - 0.5) / sigma) ** 2)
    w /= w.sum()
    indices = np.random.choice(n, p=w, size=num_states, replace=False)
    for idx in indices:
        yield reconstruct_state(game_states[idx], idx % 2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=FileType('r'))
    parser.add_argument('save_file', type=str)
    args = parser.parse_args()
    save_file = args.save_file
    lines = []
    if args.file is not None:
        lines = args.file.readlines()
    line_parser = ArgumentParser()
    line_parser.add_argument("agent1", type=agent_type)
    line_parser.add_argument("agent2", type=agent_type)
    line_parser.add_argument("num_games", type=int)
    sampled_states = []
    for line in lines:
        split_args = shlex.split(line)
        args = line_parser.parse_args(split_args)
        print(f"Running {args.num_games} simulations of \"{split_args[0]}\" vs \"{split_args[1]}\"")
        for i in range(1, args.num_games + 1):
            if i % 50 == 0:
                print(f"Simulation {i}")
            game_engine = CheckersGame(args.agent1, args.agent2)
            logger = StateLogger()
            game_engine.play_game(logger)
            logger.save(save_file)
            sampled_states.extend(sample_states(logger.past_states))
            del logger
            del game_engine
    print(f'Playing out from {len(sampled_states)} sampled states')
    for state in sampled_states:
        for agent1, agent2 in ((args.agent1, args.agent2), (args.agent2, args.agent1)):
            game_engine = CheckersGame(agent1, agent2)
            game_engine.current_state = state
            logger = StateLogger()
            logger.starting_player = state.current_player_index
            game_engine.play_game(logger)
            logger.save(save_file)
            del logger
            del game_engine