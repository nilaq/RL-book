from typing import TypeVar, Tuple, Iterator, Iterable, Callable

from rl.approximate_dynamic_programming import ValueFunctionApprox, extended_vf, evaluate_mrp, NTStateDistribution, \
    value_iteration
from rl.iterate import iterate, converged
from rl.markov_decision_process import MarkovDecisionProcess, FiniteMarkovDecisionProcess
from rl.markov_process import NonTerminal, MarkovRewardProcess, State
from rl.policy import DeterministicPolicy

A = TypeVar('A')
S = TypeVar('S')

DEFAULT_TOLERANCE = 0.001


def greedy_policy_from_approx_vf(mdp: MarkovDecisionProcess[S, A], policy_vf_approx: ValueFunctionApprox[S], γ: float) \
        -> DeterministicPolicy[S, A]:
    '''Returns a greedy policy with respect to the given value function approximator'''

    def q(s: NonTerminal[S], a: A) -> (A, float):
        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(policy_vf_approx, s1)

        return mdp.step(s, a).expectation(return_)

    actions: Callable[[NonTerminal[S]], Iterable[A]] = mdp.actions

    def optimal_action(s: S) -> A:
        q_values = {a: q(NonTerminal(s), a) for a in actions(NonTerminal(s))}
        return max(q_values, key=q_values.get)

    return DeterministicPolicy(optimal_action)


def almost_equal_func_approx(
        approx_1: ValueFunctionApprox[S],
        approx_2: ValueFunctionApprox[S],
        tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Returns whether the two value function approximators are within the given tolerance of each other'''
    return approx_1.within(approx_2, tolerance)


def almost_equal_policies_and_func_approx(
        arg_1: Tuple[DeterministicPolicy[S, A], ValueFunctionApprox[S]],
        arg_2: Tuple[DeterministicPolicy[S, A], ValueFunctionApprox[S]],
        tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Returns whether the two policies are within the given tolerance of each other'''

    pi_1, f_1 = arg_1
    pi_2, f_2 = arg_2
    return almost_equal_func_approx(f_1, f_2, tolerance)  # and pi_1 == pi_2


def approximate_policy_iteration(
        mdp: MarkovDecisionProcess[S, A],
        γ: float,
        approx_0: ValueFunctionApprox[S],
        non_terminal_states_distribution: NTStateDistribution[S],
        num_state_samples: int
) -> (Iterator[ValueFunctionApprox[S]], DeterministicPolicy[S, A]):
    '''Approximating the value function (V*) of the given MDP by improving
    the policy repeatedly after approximating the value function for a policy
    '''

    def update(args: Tuple[DeterministicPolicy[S, A], ValueFunctionApprox[S]]) \
            -> (DeterministicPolicy[S, A], ValueFunctionApprox[S]):
        pi, approx_0 = args
        mrp: MarkovRewardProcess[S, A] = mdp.apply_policy(pi)
        policy_vf_approx: ValueFunctionApprox[S] = converged(
            evaluate_mrp(mrp, γ, approx_0, non_terminal_states_distribution, num_state_samples),
            done=almost_equal_func_approx
        )
        improved_pi = greedy_policy_from_approx_vf(mdp, policy_vf_approx, γ)
        return improved_pi, policy_vf_approx

    pi_0: DeterministicPolicy[S, A] = greedy_policy_from_approx_vf(mdp, approx_0, γ)

    return iterate(update, (pi_0, approx_0))


def approximate_policy_iteration_result(
        mdp: MarkovDecisionProcess[S, A],
        gamma: float,
        approx_0: ValueFunctionApprox[S],
        non_terminal_states_distribution: NTStateDistribution[S],
        num_state_samples: int
) -> Tuple[DeterministicPolicy[S, A], ValueFunctionApprox[S]]:
    return converged(approximate_policy_iteration(mdp, gamma, approx_0, non_terminal_states_distribution,
                                                  num_state_samples), done=almost_equal_policies_and_func_approx)


# main

if __name__ == '__main__':
    from rl.chapter2.simple_inventory_mrp import InventoryState
    from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
    from rl.dynamic_programming import value_iteration_result
    from rl.distribution import Categorical
    from rl.function_approx import Tabular
    from rl.iterate import last

    # Simple Inventory MDP from chapter 2
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.1

    mdp: FiniteMarkovDecisionProcess[InventoryState, int] = \
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    # Policy Iteration parameters
    gamma: float = 0.1
    approx_0: Tabular[int] = Tabular({s: 0.0 for s in mdp.non_terminal_states})
    non_terminal_states_distribution: NTStateDistribution[int] = Categorical({
        s: 1 / len(mdp.non_terminal_states) for s in mdp.non_terminal_states
    })
    num_state_samples: int = 6


    # Approximate Policy Iteration
    pi_approx, vf_approx = approximate_policy_iteration_result(mdp, gamma, approx_0, non_terminal_states_distribution,
                                                               num_state_samples)
    print('Approximate Policy Iteration')
    for s, v in vf_approx.values_map.items():
        print(f'{s.state} -> vf: {v}, policy: {pi_approx.act(s).value}')

    # Approximate Value Iteration
    vf_iter: ValueFunctionApprox[S] = converged(
        value_iteration(mdp, gamma, approx_0, non_terminal_states_distribution,
                        num_state_samples), done=almost_equal_func_approx)
    print('-----------------------')
    print('Approximate Value Iteration')
    for s, v in vf_iter.values_map.items():
        print(f'{s.state} -> vf: {v}')

    # Deterministic Policy Iteration
    vf_deter, pi_deter = value_iteration_result(mdp, gamma)
    print('-----------------------')
    print('Deterministic Policy Iteration')
    for s, v in vf_deter.items():
        print(f'{s.state} -> vf: {v}, policy: {pi_deter.action_for[s.state]}')
