import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import argparse
import shlex
from argparse import ArgumentParser, FileType

import numpy as np

from CheckersGame import CheckersGame, StateLogger, CheckersGameState, CheckerboardCodes
from agent.Agent import RandomAgent, DeterministicAgent
from agent.CheckersAlphaBetaAgent import CheckersAlphaBetaAgent
from agent.MonteCarloAgent import MonteCarloAgent
from agent.NeuralNetworkAgent import NeuralNetworkAgent
from util import board_positions


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

agent_types = {
    "random": RandomAgent,
    "alpha-beta": CheckersAlphaBetaAgent,
    "monte-carlo": MonteCarloAgent,
    "deterministic": DeterministicAgent,
    "neural-net": NeuralNetworkAgent,
}

needs_file = {
    NeuralNetworkAgent,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=FileType('r'))
    parser.add_argument('save_file', nargs='?')
    parser.add_argument("--sample-states", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    save_file = args.save_file
    lines = []
    if args.file is not None:
        lines = args.file.readlines()
    line_parser = ArgumentParser(exit_on_error=True)
    line_parser.add_argument("agent1", choices=agent_types.keys())
    line_parser.add_argument("agent2", choices=agent_types.keys())
    line_parser.add_argument("num_games", type=int)
    line_parser.add_argument("--model-file1")
    line_parser.add_argument("--model-file2")
    line_parser.add_argument("--save-file", required=save_file is None,
                             help='Name of a binary file where game states will be stored. Required if a general save '
                                  'file was not provided in script args')
    for line in lines:
        sampled_states = []
        split_args = shlex.split(line)
        try:
            line_args = line_parser.parse_args(split_args)
            for i in range(1, 3):
                agent_name = f'agent{i}'
                agent_type = getattr(line_args, agent_name)
                cls = agent_types[agent_type]
                if cls in needs_file:
                    option_name = f'model_file{i}'
                    if (option := getattr(line_args, option_name)) is None:
                        line_parser.error(f'Must specify option {option_name} when {agent_name} = {agent_type}')
                    try:
                        obj = cls(option)
                    except ValueError:
                        line_parser.error(f'Unable to open model file {option}')
                    setattr(line_args, agent_name, obj)
                else:
                    setattr(line_args, agent_name, cls())
        except SystemExit as e:
            print("Skipping line")
            continue
        save_file = args.save_file if line_args.save_file is None else line_args.save_file
        print(f"Running {line_args.num_games} simulations of \"{split_args[0]}\" vs \"{split_args[1]}\"")
        for i in range(1, line_args.num_games + 1):
            if i % 50 == 0:
                print(f"Simulation {i}")
            game_engine = CheckersGame(line_args.agent1, line_args.agent2)
            logger = StateLogger()
            game_engine.play_game(logger)
            logger.save(save_file)
            if args.sample_states:
                sampled_states.extend(sample_states(logger.past_states))
            del logger
            del game_engine
        if len(sampled_states) > 0:
            print(f'Playing out from {len(sampled_states)} sampled states')
        for state in sampled_states:
            for agent1, agent2 in ((line_args.agent1, line_args.agent2), (line_args.agent2, line_args.agent1)):
                game_engine = CheckersGame(agent1, agent2)
                game_engine.current_state = state
                logger = StateLogger()
                logger.starting_player = state.current_player_index
                game_engine.play_game(logger)
                logger.save(save_file)
                del logger
                del game_engine