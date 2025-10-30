import random
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def load_yearpredictionmsd_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, nrows=100000) #первые 10к строк

        X_df = data.iloc[:, 1:]
        y_df = data.iloc[:, 0]

        X_df = X_df.astype(float)
        y_df = y_df.astype(float)

        return X_df, y_df

    except FileNotFoundError:
        print(f"Файл не найден")
        raise
    except Exception as e:
        print(f"Ошибка при загрузке данных")
        raise

def load_and_prepare_data():
    file_path = "YearPredictionMSD.txt"

    X_df, y_df = load_yearpredictionmsd_data(file_path)

    print(f"Загружено: {X_df.shape[0]} samples")
    print(f"Количество признаков: {X_df.shape[1]}")
    print(f"Диапазон лет: {y_df.min()} - {y_df.max()}")

    return X_df, y_df

X, y = load_and_prepare_data()

n_samples = X.shape[0]

indices = np.arange(n_samples)

np.random.shuffle(indices)

X_shuffled = X.iloc[indices]
y_shuffled = y.iloc[indices]

train_size = int(n_samples * 0.8)

X_train = X_shuffled[:train_size]
X_test = X_shuffled[train_size:]

y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

regressor = LinearRegression().fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

print(f"Коэфф. детерминации обучающей выборки: {r2_score(y_train, y_train_pred):.2f}")
print(f"Коэфф. детерминации тестовой выборки: {r2_score(y_test, y_test_pred):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="black", alpha=0.6, label="Фактические значения")

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle='--', linewidth=2,
         label="Идеальный прогноз (y=x)")

plt.xlabel("Истинные значения (год)")
plt.ylabel("Предсказанные значения (год)")
plt.title("YearPredictionMSD: истинные значения и предсказанные значения")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

degrees = range(1, 3)
r2_train_list = []
r2_test_list = []

for degree in degrees:
    print(f"Обучение с полиномиальными признаками степени {degree}")
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    r2_train_list.append(r2_score(y_train, y_train_pred))
    r2_test_list.append(r2_score(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.plot(degrees, r2_train_list, marker='o', label="Train R2")
plt.plot(degrees, r2_test_list, marker='o', label="Test R2")
plt.xlabel("Степень полиномиальных признаков")
plt.ylabel("R^2")
plt.legend()
plt.grid(True)
plt.show()

degree = 2
alphas = np.logspace(-4, 3, 10)

r2_train_list = []
r2_test_list = []

for alpha in alphas:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, max_iter=10000))
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    r2_train_list.append(r2_score(y_train, y_train_pred))
    r2_test_list.append(r2_score(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, r2_train_list, marker='o', label="Train R^2")
plt.semilogx(alphas, r2_test_list, marker='o', label="Test R^2")
plt.xlabel("Коэфф. регуляризации")
plt.ylabel("R^2")
plt.title(f"Гребневая регрессия (Степень = {degree})")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

best_index = np.argmax(r2_test_list)
best_alpha = alphas[best_index]

print(f"Наилучший коэфф. регуляризации: {best_alpha:.4f}")