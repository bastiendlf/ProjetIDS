import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('red_wines.csv')

# 2.2. Preparing data
df = data.dropna()
print("Old data frame length:", len(data))
print("New data frame length:", len(df))
print("Number of rows with at least 1 NA value: ", (len(data) - len(df)))
print("We removed ", ((len(data) - len(df)) / len(data)), "% of total amount of values.")

test1 = data.isnull()
test2 = df.isnull()

test = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
plt.show()

print("Hello IDS project")
