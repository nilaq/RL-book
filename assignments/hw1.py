import itertools
import json
from typing import Mapping, Tuple
import numpy as np
from rl.markov_process import FiniteMarkovProcess, NonTerminal, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from rl.gen_utils.plot_funcs import plot_list_of_curves


# create the transition probabilities map for the snake and ladder game
# input: snake and ladder map to represent jumps
def create_tr_map(snakes: Mapping[int, int],
                  ladders: Mapping[int, int]):  # -> Mapping[int, Categorical(Mapping[int, float])]:
    tr_map = {}

    for state in range(100):
        dist = {}
        for i in range(1, 7):
            if state + i in snakes:
                dist[snakes[state + i]] = 1 / 6
            elif state + i in ladders:
                dist[ladders[state + i]] = 1 / 6
            elif state + i == 100:
                dist[100] = 1 / 6
            elif state + i > 100:
                dist[state] = (state + i - 100) / 6
            else:
                dist[state + i] = 1 / 6

        tr_map[state] = Categorical(dist)
    return tr_map


# given a transition map, add a constant reward to each state
def create_tr_reward_map(tr_map, #Mapping[int, Categorical(Mapping[int, float])],
                         reward_constant: int):  # -> Mapping[int, Categorical[Tuple[int, float]]]:
    tr_reward_map = {}
    for k1, v1 in tr_map.items():
        if isinstance(v1, Categorical) and hasattr(v1, 'probabilities'):
            prob_dict = {}
            for k2, v2 in v1.probabilities.items():
                prob_dict[(k2, reward_constant)] = v2
            tr_reward_map[k1] = Categorical(prob_dict)
    return tr_reward_map


# function to create a plot from three traces
def plot_three_traces(
        trace1: np.ndarray,
        trace2: np.ndarray,
        trace3: np.ndarray
) -> None:
    plot_list_of_curves(
        [range(len(trace1)), range(len(trace2)), range(len(trace3))],
        [trace1, trace2, trace3],
        ["r-", "b--", "g-."],
        ['trace1', 'trace2', 'trace3'],
        'step_count',
        'field',
        'Plot of three sample traces'
    )


# function to create a histogram of the number of steps given a traces array
def plot_distribution_of_steps(traces: np.ndarray) -> None:
    steps = np.array([len(trace) for trace in traces])
    hist, bin_edges = np.histogram(steps, bins=np.arange(0, max(steps) + 5, 5))
    plot_list_of_curves(
        [bin_edges[:-1]],
        [hist / 1000],
        ["r-"],
        ['Traces = 1000'],
        'time_step_count',
        'percentage',
        'Distribution of steps (bin size = 5)'
    )


# 2b) calculate expected steps for the frog game
def calculate_expected_steps(s):
    if s == 10:
        return 0
    steps = []
    for n in range(1, 10 - s + 1):
        e_x = (1 + calculate_expected_steps(s + n)) / (10 - s)
        steps.append(e_x)
    return sum(steps)


if __name__ == '__main__':
    snakes = {32: 10, 36: 6, 48: 26, 62: 18, 88: 24, 95: 56, 97: 78}
    ladders = {1: 38, 4: 14, 8: 30, 21: 42, 28: 76, 50: 67, 71: 92, 80: 99}

    # 1 c)
    tr_map = create_tr_map(snakes, ladders)

    SnakesAndLadders = FiniteMarkovProcess(tr_map)

    traces = [list(map(lambda x: x.state, list(trace))) for trace in
              itertools.islice(SnakesAndLadders.traces(SnakesAndLadders.transition_map[NonTerminal(0)]), 1000)]

    # 1 d)
    plot_three_traces(traces[0], traces[1], traces[2])
    plot_distribution_of_steps(traces)

    # 1 e)
    tr_reward_map = create_tr_reward_map(tr_map, -1)
    SnL = FiniteMarkovRewardProcess(tr_reward_map)
    print(SnL.get_value_function_vec(1)[0])

    # 2 b)
    for i in range(11):
        print("Expected steps for s = {}: {}".format(i, calculate_expected_steps(i)))

