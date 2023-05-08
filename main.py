# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from gym import spaces
from rl.core import Env
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent


# Load the dataset
df = pd.read_csv("./Data/new.csv")

# Drop irrelevant columns (such as timestamp)
df = df.drop(columns=["Timestamp"])

# Drop columns with same values throughout
dropped_cols = list(df.columns[df.nunique() == 1])
df = df.drop(columns=dropped_cols)

# Replace infinite or very large values with the maximum of the corresponding feature
df = df.replace([np.inf, -np.inf], np.nan)
max_values = df.max()
df = df.fillna(max_values)

# Separate the target variable
X = df.drop(columns=["Label"])
y = df["Label"]

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Create a recursive feature eliminator
rfe = RFE(estimator=dtc, n_features_to_select=10)

# Fit the RFE on the data
rfe.fit(X, y)

# Print the rankings of each feature
selected_features = []
for i in range(len(X.columns)):
    if rfe.ranking_[i] == 1:
        selected_features.append(X.columns[i])
        
print(f"\nTotal number of selected features: {len(selected_features)}")
print(f"Selected features: {selected_features}")

