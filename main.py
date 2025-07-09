import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings
# filter warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/train.csv')
df = df.set_index('PassengerId')

print(df.head())
print(df.shape)
df.info()


TARGET = 'Transported'
FEATURES = [col for col in df.columns if col != TARGET]

text_features = ["Cabin", "Name"]
cat_features = [col for col in FEATURES if df[col].nunique() < 25 and col not in text_features ]
cont_features = [col for col in FEATURES if df[col].nunique() >= 25 and col not in text_features ]

print(f'Number of categorical features: {len(cat_features)}')
print('Categorical features:', cat_features, '\\n')
print(f'Number of continuos features: {len(cont_features)}')
print('Continuos features:', cont_features, '\\n')
print(f'Number of text features: {len(text_features)}')
print('Text features:', text_features)

ax = df[TARGET].value_counts().plot(kind='bar', figsize=(8, 5))
for i in ax.containers:
  ax.bar_label(i)
  ax.set_xlabel("value")
  ax.set_ylabel("count")
       
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()


ax = df.loc[:, cont_features].hist(figsize=(10, 12), grid=False, edgecolor='black', linewidth=.4)
for row in ax:
  for col in row:
    for i in col.containers:
      col.bar_label(i)
      col.set_xlabel("value")
      col.set_ylabel("count")
     
plt.suptitle("Continuous features distribution")

plt.tight_layout()
plt.show()


services_features = cont_features[1:]

for feature in services_features:
    df[f'used_{feature}'] = df.loc[:, feature].apply(lambda x: 1 if x > 0 else 0)

# Correlation matrix for selected features
corr_matrix = df.loc[:, cont_features + ['CryoSleep', 'VIP', TARGET]].corr()

# Display a styled correlation matrix (optional, requires jinja2)
try:
    print(corr_matrix)
except:
    print("Install 'jinja2' to use .style on DataFrames")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]
imputer = SimpleImputer(strategy='median')
imputer.fit(df[imputer_cols])
df[imputer_cols] = imputer.transform(df[imputer_cols])
df["HomePlanet"].fillna('Gallifrey', inplace=True)
df["Destination"].fillna('Skaro', inplace=True)

df['CryoSleep_is_missing'] = df['CryoSleep'].isna().astype(int)
df['VIP_is_missing'] = df['VIP'].isna().astype(int)

print(df['CryoSleep'].value_counts())
print(df['VIP'].value_counts())

df["CryoSleep"].fillna(False, inplace=True)
df["VIP"].fillna(False, inplace=True)

df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"] = df["VIP"].astype(int)

dummies = pd.get_dummies(df.loc[:, ['HomePlanet', 'Destination']], dtype=int)
dummies

'''
Змінна dummies виглядає так:

             HomePlanet_Earth  HomePlanet_Europa  ...  Destination_Skaro  Destination_TRAPPIST-1e
PassengerId                                       ...                                            
0001_01                     0                  1  ...                  0                        1
0002_01                     1                  0  ...                  0                        1
0003_01                     0                  1  ...                  0                        1
0003_02                     0                  1  ...                  0                        1
0004_01                     1                  0  ...                  0                        1
...                       ...                ...  ...                ...                      ...
9276_01                     0                  1  ...                  0                        0
9278_01                     1                  0  ...                  0                        0
9279_01                     1                  0  ...                  0                        1
9280_01                     0                  1  ...                  0                        0
9280_02                     0                  1  ...   
'''
df = pd.concat([df, dummies], axis=1)
df.drop(columns=['HomePlanet', 'Destination'], inplace=True)

# Оскільки модель, яку ми будемо створювати, очікує на числові значення, перетворюємо цільову ознаку з бінарної на цілочисельну.

df[TARGET] = df[TARGET].astype(int)

# Оскільки наразі ми не обробляємо текстові змінні, видалимо їх.

df.drop(["Name" ,"Cabin"] , axis=1 ,inplace = True)

# Train/test split

X = df.drop(TARGET , axis =1 )
y = df[TARGET]

X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42, test_size =0.33, stratify=y)

# Для подальших коректних обрахунків транспонуємо матрицю вхідних ознак і
# вектор цільової змінної.

x_train = X_train.T
x_test = X_test.T
y_train = np.expand_dims(y_train.T, 0)
y_test = np.expand_dims(y_test.T, 0)

print('X train size', x_train.shape)
print('X test size', x_test.shape)
print('y train size', y_train.shape)
print('y test size', y_test.shape)