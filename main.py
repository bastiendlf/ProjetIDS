import pandas as pd

data = pd.read_csv('red_wines.csv')
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# 2.2. Preparing data

# Removing outliers
# plotting values (uncomment plt.show() to see plots
for e in columns:
    data.boxplot(column=e)
#     plt.show()

# Fix issues with pH (correct values are: 0<=pH=< 14.0)
data['pH'][data['pH'] > 14] = float('nan')
data['pH'][data['pH'] < 0] = float('nan')

# Removing rows with nan values
df = data.dropna()
print("Old data frame length:", len(data))
print("New data frame length:", len(df))
print("Number of rows with at least 1 NA value: ", (len(data) - len(df)))
print("We removed ", ((len(data) - len(df)) / len(data)), "% of total values amount.")

test = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
# plt.show()

print("Hello IDS project")
