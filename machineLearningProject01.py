import pandas as pd

df= pd.read_csv('Housing.csv')
print(">>>Dataset loaded successfully.")

print(">>>First 5 rows of the dataset:")
print(df.head()) 
print(" ")
print(">>>Last 5 rows of the dataset:")
print(df.tail())
print(" ")

print(">>>Summary statistics of the dataset:")
print(df.describe())

print(" ")

print(">>>Information about the dataset:")
print(df.info())

print(" ")
print(">>>Checking for missing values in each column:")
print(df.isnull().sum())
print(" ")

# Converting categorical variables to numerical format
columns_to_transform = ['mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea']
df[columns_to_transform] = df[columns_to_transform].replace({'yes': 1, 'no': 0})
# Converting 'furnishingstatus' to numerical format
df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

print(">>>Categorical variables converted to numerical format.")
print(df.head())
print(df.tail())
print(" ")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
list2 = ['area', 'price']

df[list2] = scaler.fit_transform(df[list2])
print(">>>Numerical variables 'area' and 'price' have been standardized.")
print(df.head())
print(df.tail())
print(">>>Data preprocessing completed.")
print(" ")

print(">>>cheacking the data types after preprocessing:")
print(df.dtypes)
print(" ")

core = df.corr()
print(">>>Correlation matrix of the dataset:")
print(core)
print(" ")

import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize=(10, 10))
sns.heatmap(core, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

df.hist(figsize=(10, 10),bins=10)
plt.show()

X = df.drop('price', axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(">>>Data split into training and testing sets.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print(" ")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(">>>Linear Regression model trained successfully.")
print(" ")

y_pred = model.predict(X_test)
print(">>>Predictions made on the test set.")
print(" ")

from sklearn.metrics import r2_score

lr_accuracy = r2_score(y_test, y_pred)

print(f">>>R² score of the Linear Regression model: {lr_accuracy:.4f}")

print(">>>Machine Learning project completed.")