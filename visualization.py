import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_generation import generate_data
from q_learning import get_optimal_price

def plot_performance():
    reward_history = np.load('reward_history.npy')
    avg_q_values = np.load('avg_q_values.npy')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(avg_q_values)
    plt.title('Average Q-value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')

    plt.tight_layout()
    plt.savefig('learning_performance.png')
    plt.close()

def plot_q_values():
    Q = np.load('Q_matrix.npy')
    plt.figure(figsize=(12, 8))
    plt.imshow(Q, cmap='viridis')
    plt.colorbar(label='Q-value')
    plt.title('Q-values Heatmap')
    plt.xlabel('Action (Price Bin)')
    plt.ylabel('State (Inventory Bin * 7 + Day of Week)')
    plt.savefig('q_values_heatmap.png')
    plt.close()

def analyze_price_recommendations():
    df, kbd_inventory, kbd_price = generate_data()
    Q = np.load('Q_matrix.npy')

    inventory_levels = np.linspace(0, 200, 21).astype(int)
    days = range(7)

    results = []
    for inv in inventory_levels:
        for day in days:
            price = get_optimal_price(inv, day, Q, kbd_inventory, kbd_price)
            results.append((inv, day, price))

    results_df = pd.DataFrame(results, columns=['Inventory', 'Day', 'Optimal Price'])
    summary = results_df.groupby('Inventory')['Optimal Price'].agg(['mean', 'min', 'max'])

    plt.figure(figsize=(12, 6))
    plt.plot(summary.index, summary['mean'], label='Mean')
    plt.fill_between(summary.index, summary['min'], summary['max'], alpha=0.3)
    plt.title('Optimal Price vs Inventory Level')
    plt.xlabel('Inventory')
    plt.ylabel('Optimal Price')
    plt.legend()
    plt.savefig('optimal_price_vs_inventory.png')
    plt.close()

    return summary

if __name__ == "__main__":
    plot_performance()
    plot_q_values()
    summary = analyze_price_recommendations()
    print("Visualizations created.")
    print("\nSummary of Price Recommendations:")
    print(summary)