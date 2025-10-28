"""A/B testing with bandits: Epsilon‑Greedy and Thompson Sampling
------------------------------------
- We simulate 4 ads (arms) with true average rewards `[1, 2, 3, 4]`.
- We run two classic bandit strategies to learn which ad is best:
  - Epsilon‑Greedy with epsilon that decays as epsilon_initial / t.
  - Gaussian Thompson Sampling with a known precision (tau).
- We run 20,000 trials, record which arm we pulled and the reward we saw,
  and then we visualize learning and print/save the results.

"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter



class Bandit(ABC):
    """Base interface for any bandit algorithm we plug in here.

    Subclasses should handle:
    - how to pick an arm,
    - how to update after seeing a reward,
    - how to run the full experiment loop,
    - and how to report/save results.
    """

    @abstractmethod
    def __init__(self, p):
        """Set up the bandit with the true per‑arm rewards.

        Args:
            p (list[float]): True expected reward for each arm.
        """

    @abstractmethod
    def __repr__(self):
        """Quick one‑line description used in logs/printing."""

    @abstractmethod
    def pull(self):
        """Pull an arm and return a reward (or the arm index and reward)."""

    @abstractmethod
    def update(self):
        """Adjust internal estimates after observing a reward."""

    @abstractmethod
    def experiment(self):
        """Run the experiment loop for the configured number of trials."""

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        """Save results and print a summary, so we can inspect performance.

        Returns:
            dict: Contains cumulative reward and cumulative regret.
        """

#--------------------------------------#


class Visualization():

    def plot1(self):
        """Shows the learning process and how regret builds up.

        Left: running average reward (with an optional optimal baseline).
        Right: cumulative regret. Numbers are formatted to avoid sci‑notation.
        """
        plt.figure(figsize=(12, 5))
        if hasattr(self, "history_by_algo") and self.history_by_algo:
            # Learning curves (running average reward)
            ax1 = plt.subplot(1, 2, 1)
            optimal_avg = None
            if hasattr(self, "optimal_mean"):
                optimal_avg = float(self.optimal_mean)
            for algo, hist in self.history_by_algo.items():
                rewards = np.array(hist["rewards"], dtype=float)
                running_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                ax1.plot(running_avg, label=f"{algo}")
            if optimal_avg is not None:
                ax1.axhline(optimal_avg, color='r', linestyle='--', linewidth=1.5, label='Optimal')
            ax1.set_title("Running average reward")
            ax1.set_xlabel("Trial")
            ax1.set_ylabel("Avg Reward")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax1.legend()

            # Cumulative regret (linear scale)
            ax2 = plt.subplot(1, 2, 2)
            for algo, hist in self.history_by_algo.items():
                regret = np.array(hist["regret"], dtype=float)
                cum_regret = np.cumsum(regret)
                ax2.plot(cum_regret, label=f"{algo}")
            ax2.set_title("Cumulative regret")
            ax2.set_xlabel("Trial")
            ax2.set_ylabel("Cum Regret")
            ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax2.legend()
            plt.tight_layout()
            # Save learning figure as PNG
            plt.savefig("learning_process.png", dpi=300, bbox_inches='tight')
            logger.info("Saved learning plot to learning_process.png")
        else:
            logger.warning("Visualization.plot1 called with no history_by_algo set.")

    def plot2(self):
        """Side‑by‑side comparison of cumulative reward and cumulative regret."""
        if not hasattr(self, "history_by_algo") or not self.history_by_algo:
            logger.warning("Visualization.plot2 called with no history_by_algo set.")
            return
        plt.figure(figsize=(12, 5))

        # Cumulative rewards
        ax1 = plt.subplot(1, 2, 1)
        for algo, hist in self.history_by_algo.items():
            rewards = np.array(hist["rewards"], dtype=float)
            ax1.plot(np.cumsum(rewards), label=f"{algo}")
        ax1.set_title("Cumulative rewards")
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Cum Reward")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax1.legend()

        # Cumulative regret
        ax2 = plt.subplot(1, 2, 2)
        for algo, hist in self.history_by_algo.items():
            regret = np.array(hist["regret"], dtype=float)
            ax2.plot(np.cumsum(regret), label=f"{algo}")
        ax2.set_title("Cumulative regret")
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Cum Regret")
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax2.legend()
        plt.tight_layout()
        # Save cumulative metrics figure as PNG
        plt.savefig("cumulative_metrics.png", dpi=300, bbox_inches='tight')
        logger.info("Saved cumulative metrics plot to cumulative_metrics.png")
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy multi-armed bandit with 1/t epsilon decay.

    Parameters
    ----------
    p : list[float]
        True expected rewards per arm.
    num_trials : int
        Number of pulls to execute.
    epsilon_initial : float
        Initial epsilon, decays as epsilon_initial / t.
    noise_std : float
        Observation noise standard deviation for rewards.
    """
    def __init__(self, p: List[float], num_trials: int = 20000, epsilon_initial: float = 0.1, noise_std: float = 0.1):
        self.p = list(p)
        self.true_means = np.array(p, dtype=float)
        self.num_arms = len(self.true_means)
        self.num_trials = int(num_trials)
        self.epsilon_initial = float(epsilon_initial)
        self.noise_std = float(noise_std)

        # Estimates and counts
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.value_estimates = np.zeros(self.num_arms, dtype=float)

        # Tracking
        self.chosen_arms: List[int] = []
        self.rewards: List[float] = []
        self.regret: List[float] = []
        self.cumulative_rewards: List[float] = []
        self.cumulative_regrets: List[float] = []

        # Optimal reference
        self.optimal_idx = int(np.argmax(self.true_means))

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon_initial={self.epsilon_initial}, num_arms={self.num_arms})"

    def _epsilon_at(self, t: int) -> float:
        """Compute epsilon at trial t.

        Args:
            t (int): 1-based trial index.

        Returns:
            float: Exploration probability at trial t.
        """
        return self.epsilon_initial / max(1, t)

    def _select_arm(self, t: int) -> int:
        """Select an arm using epsilon-greedy at trial t.

        Args:
            t (int): 1-based trial index.

        Returns:
            int: Index of the chosen arm.
        """
        if np.random.random() < self._epsilon_at(t):
            return int(np.random.randint(self.num_arms))
        return int(np.argmax(self.value_estimates))

    def pull(self, arm: int) -> float:
        """Pull a given arm and observe a noisy reward.

        Args:
            arm (int): Arm index to pull.

        Returns:
            float: Observed reward.
        """
        return float(np.random.normal(self.true_means[arm], self.noise_std))

    def update(self, arm: int, reward: float) -> None:
        """Update running mean estimate for an arm.

        Args:
            arm (int): Arm index to update.
            reward (float): Observed reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / n

    def experiment(self) -> None:
        """Execute the epsilon-greedy experiment for the configured trials."""
        cum_r = 0.0
        cum_g = 0.0
        for i in range(self.num_trials):
            t = i + 1
            arm = self._select_arm(t)
            reward = self.pull(arm)
            self.update(arm, reward)

            self.chosen_arms.append(arm)
            self.rewards.append(reward)
            cum_r += reward
            self.cumulative_rewards.append(cum_r)

            regret_i = self.true_means[self.optimal_idx] - self.true_means[arm]
            self.regret.append(float(regret_i))
            cum_g += float(regret_i)
            self.cumulative_regrets.append(cum_g)

    def report(self) -> Dict[str, float]:
        """Save results to CSV and log summary metrics.

        Returns:
            dict: Dictionary with keys "cum_reward" and "cum_regret".
        """
        df = pd.DataFrame({
            "Bandit": self.chosen_arms,
            "Reward": self.rewards,
            "Algorithm": ["Epsilon-Greedy"] * len(self.rewards)
        })
        df.to_csv("epsilon_greedy_results.csv", index=False)
        cum_reward = float(self.cumulative_rewards[-1]) if self.cumulative_rewards else 0.0
        cum_regret = float(self.cumulative_regrets[-1]) if self.cumulative_regrets else 0.0
        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        avg_regret = float(np.mean(self.regret)) if self.regret else 0.0
        logger.info(f"[Epsilon-Greedy] Cumulative reward={cum_reward:.2f}, Cumulative regret={cum_regret:.2f}")
        logger.info(f"[Epsilon-Greedy] Average reward={avg_reward:.4f}, Average regret={avg_regret:.4f}")
        return {"cum_reward": cum_reward, "cum_regret": cum_regret}

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Gaussian Thompson Sampling with known precision.

    Parameters
    ----------
    p : list[float]
        True expected rewards per arm.
    num_trials : int
        Number of pulls to execute.
    precision : float
        Known precision tau = 1/sigma^2 of observation noise.
    """
    def __init__(self, p: List[float], num_trials: int = 20000, precision: float = 100.0):
        self.p = list(p)
        self.true_means = np.array(p, dtype=float)
        self.num_arms = len(self.true_means)
        self.num_trials = int(num_trials)
        self.tau = float(precision)  # known precision (inverse variance)

        # Posterior parameters per arm: Normal(mu, variance=1/lambda)
        self.mu = np.zeros(self.num_arms, dtype=float)
        self.lambda_ = np.ones(self.num_arms, dtype=float)

        # Tracking
        self.chosen_arms: List[int] = []
        self.rewards: List[float] = []
        self.regret: List[float] = []
        self.cumulative_rewards: List[float] = []
        self.cumulative_regrets: List[float] = []

        # Optimal reference
        self.optimal_idx = int(np.argmax(self.true_means))

    def __repr__(self) -> str:
        return f"ThompsonSampling(precision={self.tau}, num_arms={self.num_arms})"

    def _sample_posterior(self, arm: int) -> float:
        """Sample a mean from the posterior for a specific arm.

        Args:
            arm (int): Arm index.

        Returns:
            float: Sample from N(mu[arm], 1/lambda_[arm]).
        """
        sigma = 1.0 / np.sqrt(self.lambda_[arm])
        return float(np.random.normal(self.mu[arm], sigma))

    def _choose_arm(self) -> int:
        """Choose an arm by maximizing a posterior sample across arms.

        Returns:
            int: Index of the chosen arm.
        """
        samples = [self._sample_posterior(j) for j in range(self.num_arms)]
        return int(np.argmax(samples))

    def pull(self, arm: int) -> float:
        """Pull a given arm and observe a noisy reward.

        Args:
            arm (int): Arm index to pull.

        Returns:
            float: Observed reward.
        """
        sigma = 1.0 / np.sqrt(self.tau)
        return float(np.random.normal(self.true_means[arm], sigma))

    def update(self, arm: int, reward: float) -> None:
        """Update Gaussian posterior in closed form with known precision.

        Args:
            arm (int): Arm index to update.
            reward (float): Observed reward.
        """
        # Conjugate update (known precision):
        # mu_new = (tau * x + lambda * mu_old) / (tau + lambda)
        # lambda_new = lambda_old + tau
        self.mu[arm] = (self.tau * reward + self.lambda_[arm] * self.mu[arm]) / (self.tau + self.lambda_[arm])
        self.lambda_[arm] += self.tau

    def experiment(self) -> None:
        """Execute the Thompson Sampling experiment for the configured trials."""
        cum_r = 0.0
        cum_g = 0.0
        for _ in range(self.num_trials):
            arm = self._choose_arm()
            reward = self.pull(arm)
            self.update(arm, reward)

            self.chosen_arms.append(arm)
            self.rewards.append(reward)
            cum_r += reward
            self.cumulative_rewards.append(cum_r)

            regret_i = self.true_means[self.optimal_idx] - self.true_means[arm]
            self.regret.append(float(regret_i))
            cum_g += float(regret_i)
            self.cumulative_regrets.append(cum_g)

    def report(self) -> Dict[str, float]:
        """Save results to CSV and log summary metrics.

        Returns:
            dict: Dictionary with keys "cum_reward" and "cum_regret".
        """
        df = pd.DataFrame({
            "Bandit": self.chosen_arms,
            "Reward": self.rewards,
            "Algorithm": ["Thompson Sampling"] * len(self.rewards)
        })
        df.to_csv("thompson_sampling_results.csv", index=False)
        cum_reward = float(self.cumulative_rewards[-1]) if self.cumulative_rewards else 0.0
        cum_regret = float(self.cumulative_regrets[-1]) if self.cumulative_regrets else 0.0
        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        avg_regret = float(np.mean(self.regret)) if self.regret else 0.0
        logger.info(f"[Thompson Sampling] Cumulative reward={cum_reward:.2f}, Cumulative regret={cum_regret:.2f}")
        logger.info(f"[Thompson Sampling] Average reward={avg_reward:.4f}, Average regret={avg_regret:.4f}")
        return {"cum_reward": cum_reward, "cum_regret": cum_regret}




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    # We'll plot cumulative rewards and regrets side-by-side
    viz = Visualization()
    if hasattr(comparison, "history_by_algo"):
        viz.history_by_algo = comparison.history_by_algo  # type: ignore[attr-defined]
        viz.plot2()
    else:
        logger.warning("comparison() has no history to visualize. Run experiments first.")

if __name__=='__main__':
    # Problem setup
    np.random.seed(42)  # reproducibility
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    # Logging sample
    logger.info("Starting A/B testing experiments with Epsilon-Greedy and Thompson Sampling")

    # Epsilon-Greedy (epsilon decays as 1/t). Use small observation noise to match spec
    eg = EpsilonGreedy(Bandit_Reward, num_trials=NumberOfTrials, epsilon_initial=0.1, noise_std=0.1)
    eg.experiment()
    eg_metrics = eg.report()

    # Thompson Sampling with known precision (tau). Higher tau => lower noise variance
    ts = ThompsonSampling(Bandit_Reward, num_trials=NumberOfTrials, precision=100.0)
    ts.experiment()
    ts_metrics = ts.report()

    # Combine CSVs and produce a joint CSV as requested ({Bandit, Reward, Algorithm})
    df_eg = pd.read_csv("epsilon_greedy_results.csv")
    df_ts = pd.read_csv("thompson_sampling_results.csv")
    df_all = pd.concat([df_eg, df_ts], ignore_index=True)
    df_all.to_csv("rewards_all.csv", index=False)

    # Print cumulative reward and regret
    logger.info(f"Epsilon-Greedy -> Cumulative Reward: {eg_metrics['cum_reward']:.2f} | "
                f"Cumulative Regret: {eg_metrics['cum_regret']:.2f}")
    logger.info(f"Thompson Sampling -> Cumulative Reward: {ts_metrics['cum_reward']:.2f} | "
                f"Cumulative Regret: {ts_metrics['cum_regret']:.2f}")

    # Prepare visualization histories
    history_by_algo = {
        "Epsilon-Greedy": {"rewards": eg.rewards, "regret": eg.regret},
        "Thompson Sampling": {"rewards": ts.rewards, "regret": ts.regret},
    }
    viz = Visualization()
    viz.history_by_algo = history_by_algo  # type: ignore[attr-defined]
    viz.optimal_mean = max(Bandit_Reward)  # type: ignore[attr-defined]
    viz.plot1()
    viz.plot2()

    # Also enable comparison() API
    comparison.history_by_algo = history_by_algo  # type: ignore[attr-defined]
