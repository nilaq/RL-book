import itertools
import math
from typing import Iterable, TypeVar, Iterator, Callable, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.dynamic_programming import evaluate_mrp, almost_equal_np_arrays, almost_equal_vfs
from rl.function_approx import Tabular
import rl.td_lambda as td_lambda
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite, InventoryState
from rl.iterate import converged, iterate, last
from rl.markov_process import TransitionStep, NonTerminal
from rl.monte_carlo import mc_prediction
from rl.td import td_prediction

S = TypeVar('S')


def td_lambda_prediction_tabular(
        episodes: Iterable[Iterable[TransitionStep[S]]],
        initial_vf: Dict[S, float],
        gamma: float,
        lambda_: float
) -> Iterator[Dict[S, float]]:

    vf_approx: Dict[S, float] = initial_vf
    yield vf_approx

    for episode in episodes:
        eligibility_traces: Dict[S, float] = {s: 0.0 for s in vf_approx}
        for step in episode:
            s = step.state
            s_prime = step.next_state
            r = step.reward

            v_s_prime = vf_approx.get(s_prime, 0.0)
            td_error = r + gamma * v_s_prime - vf_approx[s]
            eligibility_traces[s] += 1.0
            for state in eligibility_traces:
                vf_approx[state] += (td_error * eligibility_traces[state])
                eligibility_traces[state] *= (gamma * lambda_)
        yield vf_approx


if __name__ == '__main__':
    # initial_vf_dict = {s: 0. for s in si_mrp.non_terminal_states}
    # simple MRP from chapter 3
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0
    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    print("Value Function")
    print("--------------")
    vf_exact = si_mrp.get_value_function_vec(user_gamma)
    print(vf_exact)
    print()

    print("Dynamic Programming")
    print("--------------")
    vf = converged(evaluate_mrp(si_mrp, gamma=user_gamma), done=almost_equal_np_arrays)
    print(vf)
    print()

    start_state_dist = si_mrp.get_stationary_distribution().map(lambda s: NonTerminal(s))


    print("Monte Carlo")
    print("--------------")
    traces = itertools.islice(si_mrp.reward_traces(start_state_dist), 1000)
    approx_0 = Tabular({s: 0. for s in si_mrp.non_terminal_states})
    vf: Tabular[NonTerminal[InventoryState]] = last(mc_prediction(traces, approx_0, user_gamma))
    print(list(vf.values_map.values()))
    print()

   
    print("TD")
    print("--------------")
    approx_0 = Tabular({s: 0. for s in si_mrp.non_terminal_states})
    transitions = itertools.islice(si_mrp.simulate_reward(start_state_dist), 10000)
    vf: Tabular[NonTerminal[InventoryState]] = last(td_prediction(transitions, approx_0, user_gamma))
    print(list(vf.values_map.values()))
    print()


    # plot

    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5

    def lr_func(n: int) -> float:
        return initial_learning_rate * (1 + (n - 1) / half_life) ** -exponent

    lamda_convergences = {}
    vf = {}
    for l in np.arange(0, 1.1, 0.2):
        approx_0=Tabular(
            values_map={s: 0. for s in si_mrp.non_terminal_states},
            count_to_weight_func=lr_func
        )
        traces = itertools.islice(si_mrp.reward_traces(start_state_dist), 200)
        curtailed_traces = (itertools.islice(trace, 100) for trace in traces)
        diff = []
        for vf_current in td_lambda.td_lambda_prediction(curtailed_traces, approx_0, user_gamma, l):
            vf = vf_current
            rsme = math.sqrt(sum((vf(s) - vf_exact[j]) ** 2 for j, s in enumerate(si_mrp.non_terminal_states)) / len(si_mrp.non_terminal_states))
            diff.append(rsme)
        lamda_convergences[l] = diff

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each row of data as a separate line
    for key, row in lamda_convergences.items():
        ax.plot(row, label=f'λ = {key:.1f}')

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('RSME')
    ax.legend(title='Lambda', loc='upper right')
    ax.set_title('TD Lambda Convergence for different lambda values - first 10000 steps')
    # Show the plot
    plt.show()


    print("TD Lambda (lambda = 0.5) Own")
    print("--------------")
    traces = itertools.islice(si_mrp.reward_traces(start_state_dist), 100)
    curtailed_traces = (itertools.islice(trace, 100) for trace in traces)

    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5


    def lr_func(n: int) -> float:
        return initial_learning_rate * (1 + (n - 1) / half_life) ** -exponent


    approx_0 = {s: 0. for s in si_mrp.non_terminal_states}
    vf: Tabular[NonTerminal[InventoryState]] = last(
        td_lambda_prediction_tabular(curtailed_traces, approx_0, user_gamma, 0.5))
    print(list(vf.values()))
    print()





