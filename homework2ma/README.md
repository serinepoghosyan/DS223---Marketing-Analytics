# A/B Testing with Multi-Armed Bandits: Epsilon-Greedy vs Thompson Sampling

## Overview
This project implements a comprehensive A/B testing simulation using two classic multi-armed bandit algorithms to solve the explore-exploit dilemma. We simulate 4 advertisement options (arms) with true expected rewards [1, 2, 3, 4] and let the algorithms learn which performs best over 20,000 trials.


## Algorithms Implemented

### 1. Epsilon-Greedy Algorithm
- **Strategy**: With probability ε, explore (choose random arm); otherwise exploit (choose best-known arm)
- **Epsilon Decay**: ε(t) = ε₀/t, where ε₀ is the initial exploration rate
- **Advantages**: Simple, intuitive, guaranteed to explore all arms
- **Disadvantages**: May waste time on clearly inferior arms

### 2. Thompson Sampling (Bayesian)
- **Strategy**: Sample from posterior distributions of each arm's reward, choose arm with highest sample
- **Bayesian Updates**: Uses Gaussian-Gaussian conjugacy with known precision τ
- **Advantages**: Naturally balances exploration and exploitation, often outperforms ε-greedy
- **Disadvantages**: More complex, requires assumptions about reward distributions

## Project Structure
```
homework2ma/
├── Bandit.py              # Main implementation
├── requirements.txt       # Dependencies
├── README.md             # This file
├── Homework 2.pdf        # Assignment specification
└── example.py            # Reference implementation
```

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps
1. Clone or download this repository
2. Navigate to the project directory
3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`: Numerical computations and random sampling
- `pandas`: Data manipulation and CSV handling
- `matplotlib`: Plotting and visualization
- `loguru`: Enhanced logging with colored output

## Usage

### Basic Usage
Run the complete experiment with default parameters:
```bash
python Bandit.py
```

### What Happens When You Run It
1. **Initialization**: Sets up 4 arms with true rewards [1, 2, 3, 4]
2. **Epsilon-Greedy Experiment**: Runs 20,000 trials with ε₀=0.1, noise_std=0.1
3. **Thompson Sampling Experiment**: Runs 20,000 trials with precision τ=100
4. **Data Collection**: Records arm choices, rewards, and regrets for each trial
5. **Visualization**: Generates learning curves and comparison plots
6. **Reporting**: Saves results to CSV files and prints summary statistics

## Output Files

### CSV Files
- **epsilon_greedy_results.csv**: Trial-by-trial results for Epsilon-Greedy
- **thompson_sampling_results.csv**: Trial-by-trial results for Thompson Sampling
- **rewards_all.csv**: Combined results from both algorithms

Each CSV contains columns:
- `Bandit`: Arm index (0-3) chosen in each trial
- `Reward`: Observed reward (true reward + noise)
- `Algorithm`: Algorithm name ("Epsilon-Greedy" or "Thompson Sampling")

### Visualization Files
- **learning_process.png**: 
  - Left panel: Running average reward over time
  - Right panel: Cumulative regret (lower is better)
- **cumulative_metrics.png**:
  - Left panel: Cumulative rewards comparison
  - Right panel: Cumulative regrets comparison

## Understanding the Results

### Key Metrics
- **Cumulative Reward**: Total reward accumulated over all trials
- **Cumulative Regret**: Total opportunity cost of not choosing the optimal arm
- **Average Reward**: Mean reward per trial
- **Average Regret**: Mean regret per trial

### Expected Behavior
- **Early Trials**: Both algorithms explore, Thompson Sampling typically explores more intelligently
- **Learning Phase**: Running average reward should increase toward the optimal value (4.0)
- **Convergence**: Thompson Sampling usually achieves lower cumulative regret

### Interpreting the Plots
1. **Learning Process Plot**:
   - Rising curves indicate successful learning
   - Optimal baseline (red dashed line) shows theoretical maximum
   - Closer to optimal = better performance

2. **Cumulative Metrics Plot**:
   - Steeper reward curves = faster learning
   - Lower regret curves = better exploration-exploitation balance

## Algorithm Parameters

### Epsilon-Greedy Parameters
- `epsilon_initial`: Starting exploration rate (default: 0.1)
- `noise_std`: Observation noise standard deviation (default: 0.1)

### Thompson Sampling Parameters
- `precision`: Known precision τ = 1/σ² (default: 100.0)
- Higher precision = less noisy observations = faster learning

### Experiment Parameters
- `num_trials`: Number of trials per algorithm (default: 20,000)
- `bandit_rewards`: True expected rewards [1, 2, 3, 4]

## Customization

### Modifying Parameters
Edit the main block in `Bandit.py`:
```python
# Change epsilon decay rate
eg = EpsilonGreedy(Bandit_Reward, epsilon_initial=0.05)

# Change Thompson Sampling precision
ts = ThompsonSampling(Bandit_Reward, precision=200.0)

# Change number of trials
NumberOfTrials = 10000
```

### Adding New Algorithms
1. Inherit from the `Bandit` abstract class
2. Implement required methods: `__init__`, `__repr__`, `pull`, `update`, `experiment`, `report`
3. Add to the main experiment loop

## Reproducibility
The code uses `np.random.seed(42)` for reproducible results. To get different runs:
- Change the seed value
- Modify algorithm parameters
- Remove the seed line for truly random results

## Performance Expectations

### Typical Results (with default parameters)
- **Epsilon-Greedy**: Cumulative regret ~2000-4000
- **Thompson Sampling**: Cumulative regret ~1000-2000
- **Final Average Reward**: Both should approach 3.5-4.0

### Factors Affecting Performance
- **Noise Level**: Higher noise makes learning harder
- **Precision**: Higher precision helps Thompson Sampling
- **Epsilon Decay**: Slower decay = more exploration
- **Number of Arms**: More arms = harder exploration problem

## Theoretical Background

### Explore-Exploit Tradeoff
The fundamental challenge in bandit problems: should we explore unknown arms or exploit our current best estimate? Both algorithms address this differently:
- **Epsilon-Greedy**: Explicit exploration probability
- **Thompson Sampling**: Probabilistic exploration based on uncertainty

### Bayesian Inference
Thompson Sampling uses Bayesian updating:
- Prior: Initial beliefs about arm rewards
- Likelihood: Observed rewards
- Posterior: Updated beliefs after each observation

### Regret Analysis
Regret measures opportunity cost:
- **Instantaneous Regret**: r*(t) - r(t), where r* is optimal reward
- **Cumulative Regret**: Sum of instantaneous regrets
- **Goal**: Minimize cumulative regret over time

## Extensions & Future Work

### Possible Improvements
1. **Contextual Bandits**: Use additional features for arm selection
2. **Non-stationary Rewards**: Handle changing reward distributions
3. **Multiple Objectives**: Optimize multiple metrics simultaneously
4. **Real Data**: Apply to actual A/B testing scenarios

---
*Last updated: Fall 2024*