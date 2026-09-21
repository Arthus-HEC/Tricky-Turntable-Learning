# Tricky Turntable: Game Theory and Adaptive Agents

A small project inspired by **Tricky Turntable** from *Super Mario Party Jamboree*.

The goal is simple: start from a game that can be solved analytically, then study what happens when an agent has to **learn how its opponent behaves**.

## 1. Game

Each team chooses an action

```text
0 = press no button
1 = press one button
2 = press two buttons
```

The final position depends on the total number of activated buttons modulo 4.

I use the reward vector

```text
(10, 5, 2, -1)
```

where `-1` represents the Bob-omb.

For Team A,

```text
u_A(x, y) = c[(x + y) mod 4]
```

which gives the payoff matrix

```text
[[10,  5,  2],
 [ 5,  2, -1],
 [ 2, -1, 10]]
```

In the benchmark case, the equilibrium strategy after eliminating dominated actions is

```text
P(action 0) = 0.5
P(action 1) = 0
P(action 2) = 0.5
```

with expected payoff `6`.

## 2. What is compared?

The code compares three simple learners.

**Bayesian marginal model** learns only the opponent's overall action frequencies `P(y)`.

**Bayesian conditional model** learns `P(y_t | x_(t-1))`, so it can detect that the opponent reacts to our previous action.

**Q-learning** learns values `Q(state, action)` directly from rewards, without explicitly modelling the opponent.

Two reactive opponents are used:

```text
Mirror: copies our previous action 90% of the time
Switch: usually maps 0 -> 2, 2 -> 0, and 1 -> 1
```

The main result is that the **state-conditioned Bayesian model reaches near-oracle performance much faster than tabular Q-learning** when the opponent structure is simple enough to model correctly.

The larger benchmark used for the project is saved in `results/final_results.csv`.

## 3. Concept drift

The second experiment changes the opponent strategy halfway through the interaction:

```text
before t = 500: (0.8, 0.0, 0.2)
after  t = 500: (0.2, 0.0, 0.8)
```

A Bayesian model that keeps its full history adapts slowly because old observations remain in the posterior.

A sliding-window Bayesian model only keeps recent observations, so it reacts much faster to the regime change.

In the larger benchmark, the sliding-window learner adapted in about **60 rounds**, versus about **315 rounds** for the model-free learner under the chosen specification.

## 4. Run the code

Install the two dependencies:

```bash
pip install -r requirements.txt
```

Run the reactive-opponent experiment:

```bash
python main.py
```

Run the concept-drift experiment:

```bash
python concept_drift.py
```

The scripts save their figures and small result tables in `results/`.

## 5. Repository structure

```text
tricky-turntable-learning/
├── README.md
├── main.py
├── concept_drift.py
├── requirements.txt
└── results/
    ├── final_results.csv
    ├── final_benchmark.png
    ├── concept_drift_results.csv
    └── concept_drift_adaptation.png
```

The code is intentionally kept simple: plain NumPy, explicit loops, and no machine-learning framework.
