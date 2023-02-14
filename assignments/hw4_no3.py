import itertools
from typing import Iterable, Tuple

import numpy as np
from numpy.random import poisson, lognormal

from rl.chapter9.order_book import OrderBook, DollarsAndShares, PriceSizePairs
from rl.distribution import Distribution, SampledDistribution
from rl.markov_process import MarkovProcess, State, NonTerminal


# Number 3

# instantiate a Markov Process that acts as a simulator of Order Book Dynamics
# and then run the simulation

class OrderBookMarkovProcess(MarkovProcess[OrderBook]):
    def __init__(self, order_book: OrderBook):
        self.order_book = order_book

    def transition(self, state: State[OrderBook]) -> Distribution[State[OrderBook]]:

        def simulate_market_orders(buy_or_sell: str) -> int:
            if (buy_or_sell != 'buy') and (buy_or_sell != 'sell'):
                raise ValueError('buy_or_sell must be either "buy" or "sell"')

            order_book = state.state
            # calculate arrival rate of market orders
            if buy_or_sell == 'buy':
                sell_limit_order_depth = sum([DaS.shares for DaS in order_book.ascending_asks])
                arrival_rate = 10 * order_book.market_depth() / (sell_limit_order_depth + 1)
            else:
                buy_limit_order_depth = sum([DaS.shares for DaS in order_book.descending_bids])
                arrival_rate = 10 * order_book.market_depth() / (buy_limit_order_depth + 1)
            # get number of market orders by Poisson distribution
            number_of_market_orders = poisson(max(arrival_rate, 2))

            # simulate market orders with a normal distribution depending on the mid-price
            total_orders = sum([np.random.normal(order_book.market_depth() * 0.1, order_book.bid_ask_spread() * 0.1)
                                for _ in range(number_of_market_orders)])
            if buy_or_sell == 'buy':
                total_orders = min(total_orders, sell_limit_order_depth * 0.8)
            else:
                total_orders = min(total_orders, buy_limit_order_depth * 0.8)
            return int(total_orders)

        def simulate_limit_order(buy_or_sell: str, order_book_after_mos: OrderBook, new_order_book: OrderBook) -> OrderBook:
            if (buy_or_sell != 'buy') and (buy_or_sell != 'sell'):
                raise ValueError('buy_or_sell must be either "buy" or "sell"')

            # calculate arrival rate of limit orders
            sell_depth = sum([DaS.shares for DaS in order_book_after_mos.ascending_asks])
            buy_depth = sum([DaS.shares for DaS in order_book_after_mos.descending_bids])
            if buy_or_sell == 'buy':
                arrival_rate = 15 * sell_depth / (sell_depth + buy_depth)
            else:
                arrival_rate = 15 * buy_depth / (sell_depth + buy_depth)
            # get number of limit orders by Poisson distribution
            number_of_limit_orders = poisson(arrival_rate)

            # randomly simulate a new mid price
            new_mid_price = np.random.normal(order_book_after_mos.mid_price(), 2)
            # simulate price order pairs
            for _ in range(number_of_limit_orders):
                quantity = int(np.random.normal(0.05 * order_book_after_mos.market_depth(), order_book_after_mos.bid_ask_spread() * 0.1))
                if buy_or_sell == 'buy':
                    price = np.random.normal(new_mid_price, 0.03 * new_mid_price) - lognormal(1, 0.4)
                    d_s, new_order_book = new_order_book.buy_limit_order(int(price), quantity)
                else:
                    price = np.random.normal(new_mid_price, 0.03 * new_mid_price) + lognormal(1, 0.4)
                    d_s, new_order_book = new_order_book.sell_limit_order(int(price), quantity)

            return new_order_book

        def next_order_book() -> OrderBook:
            order_book = state.state
            # randomly choose quantities of buy market ordersd
            buy_shares = simulate_market_orders('buy')
            d_s, new_order_book = order_book.buy_market_order(buy_shares)

            # randomly choose quantities of sell market orders
            sell_shares = simulate_market_orders('sell')
            d_s, order_book_after_mos = new_order_book.sell_market_order(sell_shares)

            # simulate buy limit orders
            new_order_book = simulate_limit_order('buy', order_book_after_mos, order_book_after_mos)
            
            # simulate sell limit orders
            new_order_book = simulate_limit_order('sell', order_book_after_mos, new_order_book)

            return NonTerminal(new_order_book)

        return SampledDistribution(next_order_book)


# main
if __name__ == '__main__':
    # create order book that resembles the one in the book
    # noinspection DuplicatedCode
    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    # instantiate a Markov Process that acts as a simulator of Order Book Dynamics
    mp = OrderBookMarkovProcess(ob0)
    for state in itertools.islice(mp.simulate(mp.transition(NonTerminal(ob0))), 10):
        order_book = state.state
        #order_book.pretty_print_order_book()
        order_book.display_order_book()
