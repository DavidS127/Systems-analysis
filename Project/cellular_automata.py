import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("Project/models/plots/cellular", exist_ok=True)


class CellularAutomata:
    def __init__(self, size=30, infection_rate=0.25, noise=0.05):
        self.size = size
        self.grid = np.random.randint(0, 2, (size, size))  # Infected = 1
        self.infection_rate = infection_rate
        self.noise = noise

    def step(self):
        new_grid = self.grid.copy()

        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.grid[max(0, i - 1):i + 2,
                                      max(0, j - 1):j + 2]

                infected_neighbors = neighbors.sum() - self.grid[i, j]

                # Infection Rule
                prob = infected_neighbors * self.infection_rate

                if np.random.rand() < prob + self.noise:
                    new_grid[i, j] = 1

        self.grid = new_grid

    def run(self, steps=20):
        snapshots = []
        for t in range(steps):
            self.step()
            snapshots.append(self.grid.copy())
        return snapshots


def simulate_cellular_automata():
    CA = CellularAutomata(size=25, infection_rate=0.2, noise=0.03)
    snapshots = CA.run(steps=25)

    # Final Plot heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(snapshots[-1], cmap="Reds")
    plt.title("Final State of Cellular Automata")
    plt.tight_layout()
    plt.savefig("Project/models/plots/cellular/final_state.png")
    plt.close()

    return snapshots
