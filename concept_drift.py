import csv
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Game and regime change
# ------------------------------------------------------------
REWARDS = np.array([10.0, 5.0, 2.0, -1.0])
ACTIONS = np.array([0, 1, 2])

PAYOFF = np.array([
    [REWARDS[(x + y) % 4] for y in ACTIONS]
    for x in ACTIONS
])

# Before t=500, the opponent mostly plays 0.
# After t=500, it mostly plays 2.
POLICY_BEFORE = np.array([0.8, 0.0, 0.2])
POLICY_AFTER = np.array([0.2, 0.0, 0.8])
CHANGE_POINT = 500


def choose_best_action(counts):
    """Best response to a Dirichlet posterior mean."""
    belief = counts / counts.sum()
    expected_payoff = PAYOFF @ belief
    return int(np.argmax(expected_payoff))


def sample_opponent(policy, rng):
    return int(rng.choice(ACTIONS, p=policy))


# ------------------------------------------------------------
# 2. Full-history Bayesian learner
# ------------------------------------------------------------
def full_history_bayes(n_steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.ones(3)  # Dirichlet(1,1,1)

    actions = []
    rewards = []

    for t in range(n_steps):
        action = choose_best_action(counts)
        policy = POLICY_BEFORE if t < CHANGE_POINT else POLICY_AFTER
        opponent_action = sample_opponent(policy, rng)

        actions.append(action)
        rewards.append(PAYOFF[action, opponent_action])
        counts[opponent_action] += 1

    return np.array(actions), np.array(rewards)


# ------------------------------------------------------------
# 3. Sliding-window Bayesian learner
# ------------------------------------------------------------
def sliding_window_bayes(n_steps=1000, window=100, seed=1):
    rng = np.random.default_rng(seed)

    # We keep only the most recent opponent actions.
    history = deque()
    counts = np.ones(3)  # Dirichlet(1,1,1) prior + window counts

    actions = []
    rewards = []

    for t in range(n_steps):
        action = choose_best_action(counts)
        policy = POLICY_BEFORE if t < CHANGE_POINT else POLICY_AFTER
        opponent_action = sample_opponent(policy, rng)

        actions.append(action)
        rewards.append(PAYOFF[action, opponent_action])

        # Add the new observation. If the window is full, forget the oldest one.
        if len(history) == window:
            old_action = history.popleft()
            counts[old_action] -= 1

        history.append(opponent_action)
        counts[opponent_action] += 1

    return np.array(actions), np.array(rewards)


# ------------------------------------------------------------
# 4. Constant-step Q-learning
# ------------------------------------------------------------
def q_learning(n_steps=1000, seed=2):
    rng = np.random.default_rng(seed)

    q = np.zeros(3)
    learning_rate = 0.08
    epsilon = 0.05

    actions = []
    rewards = []

    for t in range(n_steps):
        if rng.random() < epsilon:
            action = int(rng.choice(ACTIONS))
        else:
            action = int(np.argmax(q))

        policy = POLICY_BEFORE if t < CHANGE_POINT else POLICY_AFTER
        opponent_action = sample_opponent(policy, rng)
        reward = PAYOFF[action, opponent_action]

        # gamma = 0 here because the opponent policy is independent
        # of our previous action in this experiment.
        q[action] += learning_rate * (reward - q[action])

        actions.append(action)
        rewards.append(reward)

    return np.array(actions), np.array(rewards)


# ------------------------------------------------------------
# 5. Measure how quickly the learner adapts
# ------------------------------------------------------------
def adaptation_delay(actions, correct_action=2, threshold=0.90, window=20):
    """
    First time after the change point where at least 90% of the last
    20 actions are the new correct action.
    """
    for t in range(CHANGE_POINT + window, len(actions) + 1):
        recent = actions[t - window:t]
        if np.mean(recent == correct_action) >= threshold:
            return t - CHANGE_POINT
    return None


def average_many_runs(agent_function, n_runs=200):
    """Repeat an agent many times to get a smooth average curve."""
    all_rewards = []
    all_correct = []
    delays = []

    for seed in range(n_runs):
        actions, rewards = agent_function(seed=seed)
        all_rewards.append(rewards)
        all_correct.append(actions == 2)  # action 2 is best after the shift

        delay = adaptation_delay(actions)
        if delay is not None:
            delays.append(delay)

    return {
        "mean_reward": np.mean(all_rewards, axis=0),
        "prob_correct": np.mean(all_correct, axis=0),
        "mean_delay": np.mean(delays) if delays else np.nan,
    }


# ------------------------------------------------------------
# 6. Run
# ------------------------------------------------------------
def run_experiment():
    agents = {
        "Full-history Bayes": full_history_bayes,
        "Sliding-window Bayes": sliding_window_bayes,
        "Q-learning": q_learning,
    }

    results = {}

    for name, function in agents.items():
        results[name] = average_many_runs(function)

    print("\nConcept drift experiment\n")
    for name, result in results.items():
        print(f"{name:<22} mean adaptation delay: {result['mean_delay']:.1f} rounds")

    # Save a compact summary.
    with open("results/demo_concept_drift_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "mean_adaptation_delay"])
        for name, result in results.items():
            writer.writerow([name, result["mean_delay"]])

    # Plot the probability of choosing the new best action.
    plt.figure(figsize=(8, 4))
    for name, result in results.items():
        plt.plot(result["prob_correct"], label=name)

    plt.axvline(CHANGE_POINT, linestyle="--")
    plt.xlabel("Interaction")
    plt.ylabel("Probability of choosing action 2")
    plt.title("Adaptation after the opponent changes strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/demo_concept_drift_plot.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_experiment()
