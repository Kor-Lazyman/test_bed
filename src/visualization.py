import matplotlib.pyplot as plt
import seaborn as sns


class visualization:
    def __init__(self, inventory, item_name):
        self.inventory = inventory
        self.item_name = item_name

    def inventory_level_graph(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory.level_over_time, label='inventory_level')
        plt.xlabel('time[days]')
        plt.ylabel('inventory')
        plt.title(f'{self.item_name} inventory_level')
        plt.legend()
        plt.grid(True)
        plt.show()

    def inventory_cost_graph(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory.cost_over_time, label='inventory_cost')
        plt.xlabel('time[days]')
        plt.ylabel('inventory_cost')
        plt.title(f'{self.item_name} inventory_cost')
        plt.legend()
        plt.grid(True)
        plt.show()
