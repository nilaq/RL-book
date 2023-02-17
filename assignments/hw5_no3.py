import itertools
from typing import Iterable, TypeVar, Iterator

from rl.approximate_dynamic_programming import extended_vf
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.function_approx import Tabular
from rl.iterate import last, accumulate, converge
from rl.markov_process import TransitionStep, ReturnStep, NonTerminal
from rl.returns import returns

S = TypeVar('S')


def mc_prediction(
        trs: Iterable[Iterable[TransitionStep[S]]],
        approx_0: Tabular[S],
        gamma: float,
        episode_length_tolerance: float = 1e-6
) -> Iterator[Tabular[S]]:
    """Evaluate an MRP using the monte carlo method, simulating episodes"""

    episodes: Iterator[Iterator[ReturnStep[S]]] = \
        (returns(tr, gamma, episode_length_tolerance) for tr in trs)
    f = approx_0
    yield f

    for episode in episodes:
        f = last(f.iterate_updates(
            [(step.state, step.return_)] for step in episode
        ))
        yield f


def td_prediction(
        transitions: Iterable[TransitionStep[S]],
        approx_0: Tabular[S],
        γ: float
) -> Iterator[Tabular[S]]:
    """Evaluate an MRP using TD(0) using the given sequence of
    transitions."""

    def step(
            v: Tabular[S],
            tr: TransitionStep[S]
    ) -> Tabular[S]:
        return v.update([(
            tr.state,
            tr.reward + γ * extended_vf(v, tr.next_state)
        )])

    return accumulate(transitions, step, initial=approx_0)


if __name__ == '__main__':
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

    print("Value Function Simple Inventory MRP")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

    print("Value Function Simple Inventory MRP (MC)")
    print("--------------")
    states = si_mrp.non_terminal_states
    value_map = {s: 0.0 for s in states}
    count_map = {s: 0 for s in states}
    count_to_weight_func = lambda n: 1.0 / n
    func_approx = Tabular(value_map, count_map, count_to_weight_func)

    start_state_distribution = si_mrp.get_stationary_distribution().map(lambda s: NonTerminal(s))
    traces: Iterable[Iterable[TransitionStep[S]]] = [si_mrp.simulate_reward(start_state_distribution) for _ in
                                                     range(10000)]

    def done(f1, f2):
        return f1.within(f2, tolerance=0.0001)

    vf = last(converge(mc_prediction(traces, func_approx, user_gamma), done=done))
    for k, v in vf.values_map.items():
        print(f"{k}: {v}")

    print("Value Function Simple Inventory MRP (TD)")
    print("--------------")

    value_map = {s: 0.0 for s in states}
    count_map = {s: 0 for s in states}
    func_approx = Tabular(value_map, count_map)
    start_state_distribution = si_mrp.get_stationary_distribution().map(lambda s: NonTerminal(s))

    vf = last(itertools.islice(td_prediction(si_mrp.simulate_reward(start_state_distribution), func_approx, user_gamma),
                               1000000))
    for k, v in vf.values_map.items():
        print(f"{k}: {v}")
