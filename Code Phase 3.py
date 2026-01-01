from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. SÉLECTION DES FEATURES
features_list = train_df.columns.difference(['unit_nr', 'time_cycles', 'RUL'])
X = train_df[features_list]
y = train_df['RUL']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODÈLES
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
rmse_lr = np.sqrt(mean_squared_error(y_val, model_lr.predict(X_val)))

# Pourquoi Random Forest ? Car la dégradation est complexe et non-linéaire.
model_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model_rf.fit(X_train, y_train)
rmse_rf = np.sqrt(mean_squared_error(y_val, model_rf.predict(X_val)))

print(f"RMSE Baseline (Linéaire) : {rmse_lr:.2f}")
print(f"RMSE Avancé (Random Forest) : {rmse_rf:.2f}")