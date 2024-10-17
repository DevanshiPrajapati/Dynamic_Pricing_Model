import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def generate_data(start_date='2022-01-01', end_date='2023-12-31'):
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = {
        'date': dates,
        'day_of_week': dates.dayofweek,
        'inventory': np.random.randint(0, 200, len(dates)),
        'price': np.random.uniform(70, 130, len(dates)),
        'demand': np.random.randint(0, 30, len(dates))
    }

    df = pd.DataFrame(data)
    df['revenue'] = df['price'] * df['demand']

    kbd_inventory = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    kbd_price = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

    df['inventory_bin'] = kbd_inventory.fit_transform(df[['inventory']])
    df['price_bin'] = kbd_price.fit_transform(df[['price']])

    print("Data sample:")
    print(df.head())

    df.to_csv('pricing_data.csv', index=False)
    return df, kbd_inventory, kbd_price

if __name__ == "__main__":
    generate_data()
    print("Data generated and saved to 'pricing_data.csv'")