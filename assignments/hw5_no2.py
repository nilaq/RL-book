import itertools
import math
from dataclasses import dataclass
from typing import Tuple, Iterable, List

from rl.distribution import Categorical
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import MarkovProcess, NonTerminal, Terminal, State
from rl.policy import DeterministicPolicy, Policy

S = Tuple[float, int, float, float]
A = Tuple[float, float]


@dataclass(frozen=True)
class OBPriceSimulation(MarkovDecisionProcess):
    T: float
    delta_t: float
    gamma: float
    sigma: float
    k: float
    c: float

    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        t, I, W, S = state.state

        delta_a = (2 * I + 1) * self.gamma * self.sigma ^ 2 * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma
        delta_b = (1 - 2 * I) * self.gamma * self.sigma ^ 2 * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma

        P_a = S + delta_a
        P_b = S - delta_b

        return [(P_a, P_b)]

    def step(self, state: NonTerminal[S], action: A) \
            -> Categorical[Tuple[State[S], float]]:
        t, I, W, S = state.state
        P_a, P_b = action

        delta_a = P_a - S
        delta_b = S - P_b
        prob_sell = self.c * math.exp(-self.k * delta_a) * self.delta_t
        prob_buy = self.c * math.exp(-self.k * delta_b) * self.delta_t
        price_change = self.sigma * math.sqrt(self.delta_t)

        if t >= self.T:
            return Categorical({(Terminal((t, I, W, S)), W): 1.0})

        return Categorical({
            (NonTerminal((t + self.delta_t, I - 1, W + P_a, S + price_change)), 0): prob_sell * 0.5,
            (NonTerminal((t + self.delta_t, I - 1, W + P_a, S - price_change)), 0): prob_sell * 0.5,
            (NonTerminal((t + self.delta_t, I + 1, W - P_b, S + price_change)), 0): prob_buy * 0.5,
            (NonTerminal((t + self.delta_t, I + 1, W - P_b, S - price_change)), 0): prob_buy * 0.5,
            (NonTerminal((t + self.delta_t, I, W, S + price_change)), 0): (1 - prob_sell - prob_buy) * 0.5,
            (NonTerminal((t + self.delta_t, I, W, S - price_change)), 0): (1 - prob_sell - prob_buy) * 0.5,
        })


@dataclass(frozen=True)
class AvallanedaStoikovPolicy(Policy):
    T: float
    gamma: float
    sigma: float
    k: float

    def act(self, state: NonTerminal[S]) -> Categorical[A]:
        t, I, W, S = state.state

        delta_b = (2 * I + 1) * self.gamma * (self.sigma ** 2) * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma
        delta_a = (1 - 2 * I) * self.gamma * (self.sigma ** 2) * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma

        P_a = S + delta_a
        P_b = S - delta_b

        return Categorical({(P_a, P_b): 1.0})


@dataclass(frozen=True)
class NaivePolicy(Policy):
    optimal_spread: float

    def act(self, state: NonTerminal[S]) -> Categorical[A]:
        t, I, W, S = state.state

        P_a = S + self.optimal_spread / 2
        P_b = S - self.optimal_spread / 2

        return Categorical({(P_a, P_b): 1.0})


if __name__ == '__main__':
    T = 1
    delta_t = 0.005
    gamma = 0.1
    sigma = 2
    k = 1.5
    c = 140

    ob_sim = OBPriceSimulation(T, delta_t, gamma, sigma, k, c)

    # start state
    t_0 = 0
    I_0 = 0
    W_0 = 0
    S_0 = 100

    start_state_distribution = Categorical({NonTerminal((t_0, I_0, W_0, S_0)): 1})

    as_policy = AvallanedaStoikovPolicy(T, gamma, sigma, k)

    # Plot a single simulation trace with the Avallaneda-Stoikov policy
    ts: List[float] = []
    prices: List[Tuple[float, float, float]] = []
    wealth_inventory: List[Tuple[float, float, float]] = []
    for step in ob_sim.simulate_actions(start_state_distribution, as_policy):
        t, I, W, S = step.state.state
        P_a, P_b = step.action
        ts.append(t)
        prices.append((P_a, S, P_b))
        wealth_inventory.append((W, I * S, W + I * S))

    # plots the wealth and prices for a single simulation trace
    ''' 
    plot_list_of_curves(
        [ts, ts, ts],
        [[wealth_inventory[i][j] for i in range(len(wealth_inventory))] for j in range(len(wealth_inventory[0]))],
        ["r-", "b-", "g-"],
        ['trading PnL', 'inventory * mid price', 'Total wealth: PnL + inventory * mid price'],
        'time step',
        'wealth',
        'Single Simulation Trace: Avallaneda-Stoikov Policy'
    )

    plot_list_of_curves(
        [ts, ts, ts],
        [[prices[i][j] for i in range(len(prices))] for j in range(len(prices[0]))],
        ["r-", "b-", "g-"],
        ['ask price', 'OB mid price', 'bid price'],
        'time step',
        'price',
        'Single Simulation Trace: Avallaneda-Stoikov Policy'
    )


    '''

    avg_spreads = []
    account_balances_incl_inventory = []
    for _ in range(10000):
        spreads = []
        abii = []
        for step in ob_sim.simulate_actions(start_state_distribution, as_policy):
            t, I, W, S = step.state.state
            P_a, P_b = step.action
            spread = P_a - P_b
            spreads.append(spread)
            abii.append(W + I * S)
        avg_spread = sum(spreads) / len(spreads)
        avg_spreads.append(avg_spread)
        account_balances_incl_inventory.append(abii)

    average_spread = sum(avg_spreads) / len(avg_spreads)
    print(f"Average spread: {average_spread}")

    naive_policy = NaivePolicy(average_spread)

    ts: List[float] = []
    prices: List[Tuple[float, float, float]] = []
    wealth_inventory: List[Tuple[float, float, float]] = []
    for step in ob_sim.simulate_actions(start_state_distribution, naive_policy):
        t, I, W, S = step.state.state
        P_a, P_b = step.action
        ts.append(t)
        prices.append((P_a, S, P_b))
        wealth_inventory.append((W, I * S, W + I * S))

    '''
    plot_list_of_curves(
        [ts, ts, ts],
        [[wealth_inventory[i][j] for i in range(len(wealth_inventory))] for j in range(len(wealth_inventory[0]))],
        ["r-", "b-", "g-"],
        ['trading PnL', 'inventory * mid price', 'Total wealth: PnL + inventory * mid price'],
        'time step',
        'wealth',
        'Single Simulation Trace: Naive Policy'
    )

    plot_list_of_curves(
        [ts, ts, ts],
        [[prices[i][j] for i in range(len(prices))] for j in range(len(prices[0]))],
        ["r-", "b-", "g-"],
        ['ask price', 'OB mid price', 'bid price'],
        'time step',
        'price',
        'Single Simulation Trace: Naive Policy'
    )
    '''
    account_balances_incl_inventory_n = []
    for _ in range(10000):
        abii = []
        for step in ob_sim.simulate_actions(start_state_distribution, naive_policy):
            t, I, W, S = step.state.state
            P_a, P_b = step.action
            abii.append(W + I * S)
        account_balances_incl_inventory_n.append(abii)

    avg_wealth_as = [
        sum([account_balances_incl_inventory_n[i][j] for i in range(len(account_balances_incl_inventory_n))]) /
        len(account_balances_incl_inventory_n) for j in range(len(account_balances_incl_inventory_n[0]))]
    avg_wealth_naive = [
        sum([account_balances_incl_inventory[i][j] for i in range(len(account_balances_incl_inventory))]) /
        len(account_balances_incl_inventory) for j in range(len(account_balances_incl_inventory[0]))]

    plot_list_of_curves(
        [ts, ts],
        [avg_wealth_as, avg_wealth_naive],
        ["r-", "b-"],
        ['Avallaneda-Stoikov', 'Naive'],
        'time step',
        'wealth',
        'Average wealth over 10000 simulation traces'
    )
