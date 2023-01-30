import copy
import math
from typing import Tuple, Mapping, Iterable, TypeVar

from rl.distribution import Categorical
from rl.markov_process import NonTerminal, S, StateReward
from rl.policy import FinitePolicy

A = TypeVar('A')
ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[NonTerminal[S], ActionMapping[A, S]]


class WagesUtilityMaximization:

    def __init__(self, wages, offer_probabilities, discount_rate, alpha):
        self.wages = wages
        self.offer_probabilities = offer_probabilities
        self.discount_rate = discount_rate
        self.alpha = alpha

        self.actions = ['accept', 'deny']
        self.employment_status = ['employed', 'unemployed']
        self.non_terminal_states: Iterable[NonTerminal[Tuple[str, float]]] = [NonTerminal((status, wage))
                                                                               for status in self.employment_status for
                                                                               wage in self.wages[1:]]
        # specify reward function
        self.reward_function_vector = {
            (NonTerminal(s.state), a): (
                math.log(self.wages[0]) if a == 'deny' and s.state[0] == 'unemployed' else math.log(s.state[1])
            ) for s in self.non_terminal_states for a in self.actions
        }

        # calculate transition probabilities
        self.tr_map: Mapping[
            Tuple[NonTerminal[Tuple[str, float]], A], Categorical[NonTerminal[Tuple[str, float]]]] = {}
        for s in self.non_terminal_states:
            for a in self.actions:
                if s.state[0] == 'unemployed' and a == 'deny':
                    self.tr_map[(s, a)] = Categorical({
                        NonTerminal(('unemployed', self.wages[i + 1])): self.offer_probabilities[i] for i in
                        range(len(self.wages[1:]))
                    })
                else:  # if employed or accept offer
                    dist = {NonTerminal(('unemployed', self.wages[i + 1])): self.alpha * self.offer_probabilities[i] for i in
                            range(len(self.wages[1:]))}
                    dist[NonTerminal(('employed', s.state[1]))] = 1 - self.alpha
                    self.tr_map[(s, a)] = Categorical(dist)

        # initialize value function
        self.value_function_vector: Mapping[NonTerminal[Tuple[str, float]], float] = {
            NonTerminal(s.state): 0 for s in self.non_terminal_states
        }

        # initialize policy
        self.policy: FinitePolicy[NonTerminal[Tuple[str, float]], A] = FinitePolicy({
            s: max(self.actions, key=lambda a: self.reward_function_vector[(s, a)]) for s in self.non_terminal_states
        })

    def calculate_state_action_function(self) -> Mapping[NonTerminal, Mapping[A, float]]:
        """Calculate the state-action function for each state and policy."""
        state_action_values = {}
        for state in self.non_terminal_states:
            action_values = {}
            for action in self.actions:
                reward = self.reward_function_vector[(state, action)]
                future_reward = [
                    prob * self.value_function_vector[s_prime]
                    for s_prime, prob in self.tr_map[(state, action)]
                ]
                action_values[action] = reward + self.discount_rate * sum(future_reward)
            state_action_values[state] = action_values
        return state_action_values

    def update_value_function_and_policy(self, state_action_values: Mapping[NonTerminal, Mapping[A, float]]) -> None:
        """Calculate the optimal value for each state"""
        for state, action_values in state_action_values.items():
            self.value_function_vector[state] = max(action_values.values())
            self.policy.policy_map[state] = 'none' if state.state[0] == 'employed' \
                else max(action_values, key=action_values.get)

    def value_iteration(self, tolerance: float) -> Mapping[NonTerminal[Tuple[str, float]], float]:
        """Calculate the value function for each state using value iteration."""
        while True:
            v_i = copy.deepcopy(self.value_function_vector)
            self.update_value_function_and_policy(self.calculate_state_action_function())
            v_i_plus_1 = self.value_function_vector

            difference = max([abs(v_i_plus_1[state] - v_i[state]) for state in self.non_terminal_states])
            if difference < tolerance:
                break
        return self.value_function_vector, self.policy


if __name__ == '__main__':
    # input values
    wages = [150, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # wages w0 - w10
    offer_probabilities = [0.2, 0.05, 0.1, 0.15, 0.02, 0.08, 0.16, 0.04, 0.07, 0.13]  # offer probs for each wage
    discount_rate = 0.3
    alpha = 0.0001
    convergence_tolerance = 0.001

    # create instance of WagesUtilityMaximization
    wum = WagesUtilityMaximization(wages, offer_probabilities, discount_rate, alpha)
    value_vector, policy = wum.value_iteration(convergence_tolerance)
    for state, value in value_vector.items():
        print(f'{state.state}: {value} -> {policy.policy_map[state]}')
