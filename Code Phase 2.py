from sklearn.preprocessing import MinMaxScaler

# On repart du DataFrame 'df' chargé en Phase 1 (avec les noms sensor_1, op_setting_1, etc.)
train_df = df.copy()

# ==============================================================================
# 2.1 CALCUL DU RUL (Cible décroissante)
# ==============================================================================
# Étape A : On trouve le cycle maximum pour chaque moteur
max_cycle_per_engine = train_df.groupby('unit_nr')['time_cycles'].transform('max')

# Étape B : FORMULE : RUL = Max - Actuel
# Cela crée un compte à rebours (ex: 192, 191, 190... jusqu'à 0)
train_df['RUL'] = max_cycle_per_engine - train_df['time_cycles']

# --- VÉRIFICATION DU CALCUL ---
print("Vérification du RUL décroissant (Moteur 1) :")
print(train_df[train_df['unit_nr'] == 1][['unit_nr', 'time_cycles', 'RUL']].head(10))
# On doit voir le RUL diminuer à chaque ligne.

# ==============================================================================
# 2.2 STRATÉGIE DE PRÉCISION : RUL TRONQUÉ (CLIPPING)
# ==============================================================================
# Pourquoi ? Un moteur sain ne montre pas de dégradation au début. 
# Si on plafonne à 125, l'IA prédira 125 pendant le début de vie, 
# puis commencera à diminuer (124, 123...) quand l'usure devient réelle.
# C'est ce qui permet d'avoir des résultats proches du réel.
train_df['RUL'] = train_df['RUL'].clip(upper=125)

# ==============================================================================
# 2.3 NETTOYAGE (Suppression des capteurs constants)
# ==============================================================================
# POURQUOI ? On utilise la liste 'constant_sensors' de la Phase 1.
# Supprimer ces colonnes évite d'apprendre des données inutiles à l'IA.
cols_to_drop = constant_sensors + ['op_setting_3'] # op_setting_3 est souvent constant aussi
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print(f"\nNettoyage terminé. Colonnes supprimées : {cols_to_drop}")

# ==============================================================================
# 2.4 NORMALISATION (Min-Max Scaling)
# ==============================================================================
# POURQUOI ? Pour mettre tous les capteurs à la même échelle (entre 0 et 1).
scaler = MinMaxScaler()
# On ne normalise pas unit_nr, time_cycles et la cible RUL
cols_to_normalize = train_df.columns.difference(['unit_nr', 'time_cycles', 'RUL'])

train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])

print("\n✔ PHASE 2 terminée : Le RUL décroissant est prêt.")
# Aperçu final
display(train_df.head())