import numpy as np
import pandas as pd
from data_generation import generate_data

def train_q_learning(df, n_episodes=5000):
    n_states = 10 * 7  # 10 inventory bins, 7 days of week
    n_actions = 10  # 10 price bins
    Q = np.zeros((n_states, n_actions))

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1

    reward_history = []
    avg_q_values = []

    for episode in range(n_episodes):
        state = np.random.randint(0, n_states)
        total_reward = 0
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.argmax(Q[state, :])
            
            next_state = np.random.randint(0, n_states)
            reward = df['revenue'].iloc[next_state]
            
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            
            total_reward += reward
            state = next_state
            done = np.random.random() < 0.1
        
        reward_history.append(total_reward)
        avg_q_values.append(np.mean(Q))

    print("Q-learning training completed.")
    return Q, reward_history, avg_q_values

def get_optimal_price(inventory, day_of_week, Q, kbd_inventory, kbd_price):
    inventory_bin = kbd_inventory.transform([[inventory]])[0][0]
    state = int(inventory_bin * 7 + day_of_week)
    action = np.argmax(Q[state, :])
    price_range = kbd_price.inverse_transform([[action]])[0][0]
    return (price_range * (130 - 70) / 10) + 70  # Scale back to original price range

if __name__ == "__main__":
    df, kbd_inventory, kbd_price = generate_data()
    Q, reward_history, avg_q_values = train_q_learning(df)
    
    np.save('Q_matrix.npy', Q)
    np.save('reward_history.npy', reward_history)
    np.save('avg_q_values.npy', avg_q_values)
    
    print("Q-learning results saved.")