import pandas as pd

path = '../input/train.csv'

df = pd.read_csv(path)
df_part = df[:1000]
df_part.to_csv('train_part.csv')

