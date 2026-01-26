import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = sns.load_dataset('iris')


df.iloc[5:10, 2] = np.nan

print("Before handling missing values:")
print(df.isnull().sum())


df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nAfter handling missing values:")
print(df.isnull().sum())


encoder = LabelEncoder()
df['species_encoded'] = encoder.fit_transform(df['species'])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, :-2])

scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-2])

print("\nScaled Features:")
print(scaled_df.head())


plt.hist(df['sepal_length'], bins=10)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()


sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.show()
