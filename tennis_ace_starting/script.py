#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_df = pd.read_csv('tennis_stats.csv')
print(tennis_df.columns)

# perform exploratory analysis here:
# plt.scatter(
#   tennis_df['Wins'],
#   tennis_df['Aces']
# )
# plt.show()

## perform single feature linear regressions here:
# X = tennis_df['Wins']
# X = X.values.reshape(-1, 1)

# y = tennis_df['Aces']
# y = y.values.reshape(-1, 1)

# regr = LinearRegression()
# regr.fit(X, y)
# y_predict = regr.predict(X)

# plt.scatter(X, y)
# plt.scatter(X, y_predict)
# plt.show()

## perform two feature linear regressions here:
x = tennis_df[[
  'Aces',
  'BreakPointsOpportunities',
]]
y = tennis_df[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(
  x, y, train_size = 0.8, test_size = 0.2, random_state = 6,
)

lm = LinearRegression()
model = lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

print("Train Score:")
print(lm.score(x_train, y_train))

print("Test Score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)

plt.show()

## perform multiple feature linear regressions here:
x = tennis_df[[
  'Aces',
  'BreakPointsOpportunities',
  'ReturnGamesWon',
]]
y = tennis_df[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(
  x, y, train_size = 0.8, test_size = 0.2, random_state = 6,
)

lm = LinearRegression()
model = lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

print("Train Score:")
print(lm.score(x_train, y_train))

print("Test Score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)

plt.show()
