from random import random

import numpy as np
from scipy.stats import norm

class LSPI:
    def __init__(self, state_dim, action_dim, gamma=0.95, num_iterations=100, num_samples=10000, epsilon=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.epsilon = epsilon

        self.weights = np.zeros((self.state_dim,))

    def train(self, policy, feature_fn, reward_fn, transition_fn):
        for i in range(self.num_iterations):
            X = np.zeros((self.num_samples, self.state_dim))
            y = np.zeros((self.num_samples,))

            # Collect samples
            for j in range(self.num_samples):
                s = np.random.uniform(low=0.1, high=1.0)  # underlying asset price
                X[j] = feature_fn(s)
                a = policy(X[j], self.weights)
                s_next, r = transition_fn(s, a, reward_fn)
                y[j] = r + self.gamma * np.max(policy(feature_fn(s_next), self.weights))

            # Perform least squares regression
            A = np.dot(X.T, X) + self.epsilon * np.identity(self.state_dim)
            b = np.dot(X.T, y)
            self.weights = np.linalg.solve(A, b)

    def value_function(self, feature_fn, policy):
        def v(s):
            return np.dot(feature_fn(s), self.weights)

        return v


def call_option_payoff(s, k):
    return max(s - k, 0)


def transition_fn(s, a, reward_fn):
    r = reward_fn(s, a)
    s_next = np.random.normal(loc=s + a * s, scale=0.1 * s)
    return s_next, r


def american_option_price(s0, k, r, sigma, t, num_iterations=100, num_samples=10000):
    dt = t / num_iterations
    gamma = np.exp(-r * dt)

    def feature_fn(s):
        return np.array([1, s, s**2])

    def reward_fn(s, a):
        return -call_option_payoff(s, k) + gamma * call_option_payoff(s + a * s, k)

    def policy(feature, weights):
        a = -feature[1] / (2 * feature[2]) if feature[2] > 0 else 0
        return a

    lspi = LSPI(state_dim=3, action_dim=1, gamma=gamma, num_iterations=10, num_samples=num_samples, epsilon=0.01)
    lspi.train(policy, feature_fn, reward_fn, transition_fn)

    v = lspi.value_function(feature_fn, policy)

    return call_option_payoff(s0, k) + np.exp(-r * t) * np.mean([v(np.random.normal(loc=s0, scale=sigma*np.sqrt(t))) for _ in range(num_samples)])


import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size, gamma, alpha, epsilon, epsilon_min, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma    # discount rate
        self.alpha = alpha    # learning rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def american_option_price_DQN(S, K, r, sigma, T, n, is_call=True):
    """
    Calculates the price of an American Option using the Deep-Q-Learning algorithm.

    Parameters:
    S (float): the initial asset price
    K (float): the strike price
    r (float): the risk-free interest rate
    sigma (float): the volatility of the underlying asset
    T (float): the time to maturity in years
    n (int): the number of time steps to take
    is_call (bool): True for a call option, False for a put option

    Returns:
    float: the estimated price of the option
    """

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p

    state_size = 2  # current asset price and time step
    action_size = 2  # hold or exercise
    agent = DQNAgent(state_size, action_size, gamma=0.95, alpha=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    for t in range(n, -1, -1):
        for price in S * u**np.arange(t, -1, -1) * d**(np.arange(0, t+1)):
            state = np.array([[price, t]])
            action = agent.act(state)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 10
    spot_price_frac_val: float = 0.02

    option_price_LSPI = american_option_price(spot_price_val, strike, rate_val, vol_val, expiry_val)
    option_price_DQN = american_option_price_DQN(spot_price_val, strike, rate_val, vol_val, expiry_val, num_steps_val)

    print(f"LSPI: The price of the American call option is: {option_price_LSPI:.2f}")
    print(f"DQN: The price of the American call option is: {option_price_DQN:.2f}")


