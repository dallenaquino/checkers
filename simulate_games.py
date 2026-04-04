import argparse
import shlex
from argparse import ArgumentParser, FileType

from CheckersGame import CheckersGame, StateLogger
from agent.Agent import RandomAgent, DeterministicAgent
from agent.CheckersAlphaBetaAgent import CheckersAlphaBetaAgent
from agent.MonteCarloAgent import MonteCarloAgent


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=FileType('r'))
    args = parser.parse_args()
    lines = []
    if args.file is not None:
        lines = args.file.readlines()
    line_parser = ArgumentParser()
    line_parser.add_argument("agent1", type=agent_type)
    line_parser.add_argument("agent2", type=agent_type)
    line_parser.add_argument("num_games", type=int)
    for line in lines:
        split_args = shlex.split(line)
        args = line_parser.parse_args(split_args)
        print(f"Running {args.num_games} simulations of \"{split_args[0]}\" vs \"{split_args[1]}\"")
        for i in range(args.num_games):
            if (i + 1) % 50 == 0:
                print(f"Simulation {i}")
            game_engine = CheckersGame(args.agent1, args.agent2)
            logger = StateLogger()
            game_engine.play_game(logger)
            logger.save("game_states_test.bin")
            del logger
            del game_engine