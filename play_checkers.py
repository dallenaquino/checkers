import argparse
import faulthandler
from argparse import ArgumentParser

from CheckersGame import CheckersGame
from CheckersUI import CheckersUI
from agent.Agent import RandomAgent, DeterministicAgent
from agent.CheckersAlphaBetaAgent import CheckersAlphaBetaAgent
from agent.MonteCarloAgent import MonteCarloAgent
from agent.GreedyNeuralNetworkAgent import GreedyNeuralNetworkAgent
from agent.NeuralNetworkAlphaBetaAgent import NeuralNetworkAgent


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
        case "neural-net":
            return GreedyNeuralNetworkAgent('models/resnet_v0_7.keras', epsilon=0)
            # return NeuralNetworkAgent('models/resnet_v0_9.keras', epsilon=0, max_depth=3)
        case _:
            raise argparse.ArgumentTypeError(f"Unrecognized agent type: {s}")


if __name__ == "__main__":
    faulthandler.enable()
    parser = ArgumentParser()
    parser.add_argument("--agent1", default="random", type=agent_type)
    parser.add_argument("--agent2", default="random", type=agent_type)
    args = parser.parse_args()
    game_engine = CheckersGame(args.agent1, args.agent2)
    ui = CheckersUI(game_engine)
    ui.run()