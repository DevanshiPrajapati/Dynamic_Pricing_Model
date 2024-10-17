# Dynamic Pricing Model using Reinforcement Learning

This project implements a dynamic pricing model using Q-learning, a Reinforcement Learning technique. It generates synthetic pricing data, trains a Q-learning agent, and visualizes the results.

## Files

- `data_generation.py`: Generates synthetic pricing data.
- `q_learning.py`: Implements the Q-learning algorithm for price optimization.
- `visualization.py`: Creates visualizations of the learning process and price recommendations.

## Usage

1. Clone the repository
2. Generate data using 'data_generation.py'. This script creates synthetic pricing data and saves it to 'pricing_data.csv'.
3. Train the Q-learning model using 'q_learning.py'. This trains the Q-learning agent and saves the results as NumPy files.
4. Visualize results using 'visualization.py'. This creates various visualizations of the learning process and price recommendations.

## Data Generation

The `data_generation.py` script creates a dataset with the following features[1]:
- Date
- Day of week
- Inventory
- Price
- Demand
- Revenue

It also discretizes the inventory and price into bins for use in the Q-learning algorithm.

## Q-Learning

The `q_learning.py` script implements the Q-learning algorithm with the following parameters:
- Learning rate: 0.1
- Discount factor: 0.9
- Epsilon (for exploration): 0.1
- Number of episodes: 5000

The script trains the agent and saves the Q-matrix, reward history, and average Q-values.

## Visualization

The `visualization.py` script creates several plots[2]:
- Learning performance (rewards and average Q-values over episodes)
- Q-values heatmap
- Optimal price vs. inventory level

## Results

The project generates several output files:
- `pricing_data.csv`: Synthetic pricing data[1]
- `Q_matrix.npy`: Learned Q-values[3]
- `reward_history.npy`: Training reward history[3]
- `avg_q_values.npy`: Average Q-values during training[3]
- `learning_performance.png`: Plot of rewards and average Q-values[2]
- `q_values_heatmap.png`: Heatmap of learned Q-values[2]
- `optimal_price_vs_inventory.png`: Plot of optimal prices vs. inventory levels[2]

## Future Improvements

- Implement more sophisticated RL algorithms (e.g., Deep Q-Network)
- Incorporate real-world data for training and validation
- Add seasonality and trend components to the pricing model
- Develop a web interface for real-time price recommendations
