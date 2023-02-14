import itertools
from dataclasses import dataclass
from typing import Tuple, Sequence, Callable, Iterator

import numpy as np

from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from rl.distribution import SampledDistribution, Constant, Gaussian
from rl.function_approx import LinearFunctionApprox, FunctionApprox
from rl.iterate import converged
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import NonTerminal, State, Terminal
from rl.policy import Always, DeterministicPolicy


@dataclass(frozen=True)
class OptimalExcerciseAmericanOption:
    spot_price: float
    strike: float
    num_steps: int

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, bool]:

        strike: float = self.strike

        class AmericanOptionSimulator(MarkovDecisionProcess[float, bool]):

            def step(self, price: NonTerminal[float], exercise: bool) \
                    -> SampledDistribution[Tuple[State[float], float]]:
                def sr_sampler_func(price=price, exercise=exercise) -> Tuple[State[float], float]:
                    if exercise:
                        return Terminal(0.), max(strike - price.state, 0.)
                    else:
                        next_price: float = np.random.normal(price.state, 1)
                        return NonTerminal(next_price), 0.

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=200
                )

            def actions(self, price: NonTerminal[float]) -> Sequence[bool]:
                return [True, False]

        return AmericanOptionSimulator()

    def get_vf_func_approx(
            self, t: int,
            reg_coeff: float
    ) -> FunctionApprox[NonTerminal[float]]:
        return LinearFunctionApprox.create(
            feature_functions=[lambda x: x.state, lambda x: 1.],
            regularization_coeff=reg_coeff,
            direct_solve=True
        )

    def get_states_distribution(self, t: int) -> SampledDistribution[NonTerminal[float]]:
        # Since the price change of the asset is distributed as N(0,1) for every time step
        # we know that the distribution of the price change at from time 0 to time t is N(0, t)
        return SampledDistribution(lambda: NonTerminal(np.random.normal(self.spot_price, t)))

    def backward_induction_vf_and_pi(
            self,
            reg_coeff: float,
            gamma: float
    ) -> Iterator[
        Tuple[FunctionApprox[NonTerminal[float]],
        DeterministicPolicy[float, bool]]
    ]:
        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, bool],
            FunctionApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(t=i),
            self.get_vf_func_approx(
                t=i,
                reg_coeff=reg_coeff
            ),
            self.get_states_distribution(t=i)
        ) for i in range(self.num_steps + 1)]

        num_state_samples: int = 1000

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=gamma,
            num_state_samples=num_state_samples,
            error_tolerance=1e-8
        )


if __name__ == '__main__':
    optimal_ex_ao = OptimalExcerciseAmericanOption(
        spot_price=100,
        strike=100,
        num_steps=10
    )

    reg_coeff = 0.001

    it_vf: Iterator[Tuple[FunctionApprox[NonTerminal[float]], DeterministicPolicy[float, bool]]] \
        = optimal_ex_ao.backward_induction_vf_and_pi(
            reg_coeff=reg_coeff,
            gamma=1
        )

    vf_star, pi_star = converged(it_vf, done=lambda f1, f2: f1[0].within(f2[0], 0.001))

    print(vf_star)
    print(pi_star)
