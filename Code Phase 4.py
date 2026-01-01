# 1. CHARGEMENT TEST (Mêmes noms de colonnes que Phase 1)
test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=columns)
y_true = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['true_RUL'])

# 2. DERNIER CYCLE SEULEMENT
X_test = test_df.groupby('unit_nr').last().reset_index()

# 3. PRÉTRAITEMENT
X_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

# 4. PRÉDICTION ET SCORE
X_test_final = X_test[features_list]
y_pred = model_rf.predict(X_test_final)
rmse_final = np.sqrt(mean_squared_error(y_true['true_RUL'], y_pred))

print("\n" + "="*45)
print(f"RÉSULTAT FINAL RMSE SUR TEST : {rmse_final:.2f}")
print("="*45)

# 5. VISUALISATION FINALE (IA vs Réalité)
plt.figure(figsize=(12, 6))
plt.plot(y_true['true_RUL'].values, label='RUL Réel', color='blue', marker='o')
plt.plot(y_pred, label='Prédiction IA', color='red', linestyle='--', marker='x')
plt.title("Évaluation Finale : IA vs Réalité")
plt.legend(); plt.show()

# Aperçu des 5 premiers résultats
resultats = pd.DataFrame({'Moteur': X_test['unit_nr'], 'Réel': y_true['true_RUL'], 'IA': y_pred.astype(int)})
display(resultats.head())