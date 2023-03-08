from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovRewardProcess

S = Tuple[int, int]

class RandomWalkMRP2D(FiniteMarkovRewardProcess[S]):

    barrier1: int
    barrier2: int
    pUP: float
    pDOWN: float
    pLEFT: float
    pRIGHT: float

    def __init__(
        self,
        barrier1: int,
        barrier2: int,
        pUP: float,
        pDOWN: float,
        pLEFT: float,
        pRIGHT: float
    ):
        if pUP + pDOWN + pLEFT + pRIGHT != 1:
            raise ValueError("Probabilities must sum to 1")
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        self.pUP = pUP
        self.pDOWN = pDOWN
        self.pLEFT = pLEFT
        self.pRIGHT = pRIGHT
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[S, Categorical[Tuple[S, float]]]:
        d: Dict[S, Categorical[Tuple[S, float]]] = {
            (i, j): Categorical({
                ((i + 1, j), 0. if i < self.barrier1 - 1 else 1.): self.pUP,
                ((i - 1, j), 0. if i > 1 else 1.): self.pDOWN,
                ((i, j + 1), 0. if j < self.barrier2 - 1 else 1.): self.pRIGHT,
                ((i, j - 1), 0. if j > 1 else 1.): self.pLEFT
            }) for i in range(1, self.barrier1) for j in range(1, self.barrier2)
        }
        return d


if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc

    barrier: int = 10
    barrier2: int = 10
    pUP: float = 0.25
    pDOWN: float = 0.25
    pLEFT: float = 0.25
    pRIGHT: float = 0.25
    random_walk: RandomWalkMRP2D = RandomWalkMRP2D(
        barrier1=barrier,
        barrier2=barrier2,
        pUP=pUP,
        pDOWN=pDOWN,
        pLEFT=pLEFT,
        pRIGHT=pRIGHT
    )
    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=700,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )
