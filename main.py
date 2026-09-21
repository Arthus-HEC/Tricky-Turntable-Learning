import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. The game
# ------------------------------------------------------------
# Four outcomes around the turntable:
# 10 coins, 5 coins, 2 coins, and a Bob-omb (-1).
REWARDS = np.array([10.0, 5.0, 2.0, -1.0])
ACTIONS = np.array([0, 1, 2])

# Team A payoff:
# u_A(x, y) = c_(x+y mod 4)
PAYOFF = np.array([
    [REWARDS[(x + y) % 4] for y in ACTIONS]
    for x in ACTIONS
])

# The two reactive opponents used in the experiment.
# They react to OUR previous action.
def mirror_opponent(previous_action, rng, noise=0.10):
    """Copies our previous action with probability 90%."""
    n = len(previous_action)
    random_action = rng.integers(0, 3, size=n)
    follows_rule = rng.random(n) >= noise
    return np.where(follows_rule, previous_action, random_action)


def switch_opponent(previous_action, rng, noise=0.10):
    """Usually switches 0<->2 and keeps 1."""
    target = np.where(
        previous_action == 0,
        2,
        np.where(previous_action == 2, 0, 1),
    )
    n = len(previous_action)
    random_action = rng.integers(0, 3, size=n)
    follows_rule = rng.random(n) >= noise
    return np.where(follows_rule, target, random_action)


# ------------------------------------------------------------
# 2. Small helper
# ------------------------------------------------------------
def random_argmax(values, rng):
    """Argmax with random tie-breaking."""
    tiny_noise = rng.uniform(0.0, 1e-10, size=values.shape)
    return np.argmax(values + tiny_noise, axis=1)


# ------------------------------------------------------------
# 3. Bayesian learner, but WITHOUT state
# ------------------------------------------------------------
def bayes_marginal(opponent, n_runs=2000, n_steps=1000, seed=0):
    """
    Learns only the opponent's global action frequencies P(y).
    It ignores the fact that the opponent may react to our previous action.
    """
    rng = np.random.default_rng(seed)

    # Dirichlet(1,1,1) prior for each simulation.
    counts = np.ones((n_runs, 3))
    previous_action = rng.integers(0, 3, size=n_runs)
    rewards = np.zeros((n_runs, n_steps))

    for t in range(n_steps):
        belief = counts / counts.sum(axis=1, keepdims=True)
        expected_payoff = belief @ PAYOFF.T
        action = random_argmax(expected_payoff, rng)

        opponent_action = opponent(previous_action, rng)
        rewards[:, t] = PAYOFF[action, opponent_action]

        counts[np.arange(n_runs), opponent_action] += 1
        previous_action = action

    return rewards


# ------------------------------------------------------------
# 4. Bayesian learner WITH the right state
# ------------------------------------------------------------
def bayes_conditional(opponent, n_runs=2000, n_steps=1000, seed=1):
    """
    Learns P(y_t | x_(t-1)).

    There is one Dirichlet distribution for each possible previous action:
    previous action = 0, 1, or 2.
    """
    rng = np.random.default_rng(seed)

    # counts[simulation, previous_action, opponent_action]
    counts = np.ones((n_runs, 3, 3))
    previous_action = rng.integers(0, 3, size=n_runs)
    rewards = np.zeros((n_runs, n_steps))
    rows = np.arange(n_runs)

    for t in range(n_steps):
        local_counts = counts[rows, previous_action]
        belief = local_counts / local_counts.sum(axis=1, keepdims=True)

        expected_payoff = belief @ PAYOFF.T
        action = random_argmax(expected_payoff, rng)

        opponent_action = opponent(previous_action, rng)
        rewards[:, t] = PAYOFF[action, opponent_action]

        counts[rows, previous_action, opponent_action] += 1
        previous_action = action

    return rewards


# ------------------------------------------------------------
# 5. Simple tabular Q-learning
# ------------------------------------------------------------
def q_learning(opponent, n_runs=2000, n_steps=1000, seed=2):
    """
    State = our previous action.
    Action = our current action.

    This learner does NOT model the opponent explicitly.
    It learns Q(state, action) directly from rewards.
    """
    rng = np.random.default_rng(seed)

    q = np.zeros((n_runs, 3, 3))
    state = rng.integers(0, 3, size=n_runs)
    rewards = np.zeros((n_runs, n_steps))
    rows = np.arange(n_runs)

    gamma = 0.90

    for t in range(n_steps):
        # Exploration slowly decreases over time.
        epsilon = max(0.02, 0.30 * (1 - t / n_steps))

        greedy_action = random_argmax(q[rows, state], rng)
        explore = rng.random(n_runs) < epsilon
        random_action = rng.integers(0, 3, size=n_runs)
        action = np.where(explore, random_action, greedy_action)

        opponent_action = opponent(state, rng)
        reward = PAYOFF[action, opponent_action]
        rewards[:, t] = reward

        next_state = action

        # Standard Q-learning update.
        learning_rate = 0.30 / np.sqrt(1 + t / 100)
        target = reward + gamma * q[rows, next_state].max(axis=1)
        old_value = q[rows, state, action]
        q[rows, state, action] = old_value + learning_rate * (target - old_value)

        state = next_state

    return rewards


# ------------------------------------------------------------
# 6. Run the experiment
# ------------------------------------------------------------
def run_experiment():
    opponents = {
        "Mirror (90%)": mirror_opponent,
        "Switch (90%)": switch_opponent,
    }

    results = []

    for i, (name, opponent) in enumerate(opponents.items()):
        marginal = bayes_marginal(opponent, seed=100 + i)
        conditional = bayes_conditional(opponent, seed=200 + i)
        qlearn = q_learning(opponent, seed=300 + i)

        result = {
            "opponent": name,
            "bayes_marginal": marginal.mean(),
            "bayes_conditional": conditional.mean(),
            "q_learning": qlearn.mean(),
            "bayes_conditional_last_100": conditional[:, -100:].mean(),
            "q_learning_last_100": qlearn[:, -100:].mean(),
        }
        results.append(result)

    # Print a small table.
    print("\nTricky Turntable - reactive opponents\n")
    print(f"{'Opponent':<15} {'Bayes marginal':>15} {'Bayes conditional':>18} {'Q-learning':>12}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['opponent']:<15} "
            f"{r['bayes_marginal']:>15.3f} "
            f"{r['bayes_conditional']:>18.3f} "
            f"{r['q_learning']:>12.3f}"
        )

    # Save the numbers.
    with open("results/demo_reactive_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Simple plot of the final 100 interactions.
    labels = [r["opponent"] for r in results]
    bayes_values = [r["bayes_conditional_last_100"] for r in results]
    q_values = [r["q_learning_last_100"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, bayes_values, width, label="Bayes conditional")
    plt.bar(x + width / 2, q_values, width, label="Q-learning")
    plt.xticks(x, labels)
    plt.ylabel("Average payoff")
    plt.title("Final 100 interactions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/demo_reactive_plot.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Payoff matrix for Team A:")
    print(PAYOFF)
    run_experiment()
