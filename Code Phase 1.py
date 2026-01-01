import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration affichage
plt.style.use("default")
sns.set_context("notebook")

# 1. CHARGEMENT
file_path = "train_FD001.txt"
# On définit les noms qui seront utilisés dans TOUTES les phases pour éviter les erreurs
columns = (['unit_nr', 'time_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)])

df = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)

# 2. APERÇU ET STATISTIQUES (Restauré)
print("\n===== Aperçu des données =====")
display(df.head())
print("\n===== Informations générales =====")
df.info()
print("\n===== Statistiques descriptives =====")
display(df.describe())

# 3. INFORMATIONS GLOBALES (Restauré)
print("\n===== Nombre total de moteurs =====")
print(df['unit_nr'].nunique())
print("\n===== Nombre total de lignes =====")
print(len(df))

# Durée de vie maximale par moteur
engine_life = df.groupby('unit_nr')['time_cycles'].max()
print("\n===== Durée de vie des 5 premiers moteurs =====")
print(engine_life.head())

# 4. VISUALISATIONS (Restauré)
engine_id = 1
engine_data = df[df['unit_nr'] == engine_id]

# Graphique 1
plt.figure(figsize=(14, 6))
plt.plot(engine_data['time_cycles'], engine_data['sensor_2'], label='sensor_2')
plt.plot(engine_data['time_cycles'], engine_data['sensor_3'], label='sensor_3')
plt.plot(engine_data['time_cycles'], engine_data['sensor_4'], label='sensor_4')
plt.title(f"Évolution de capteurs – Moteur {engine_id}")
plt.legend(); plt.grid(True); plt.show()

# Graphique 2 (Multi-capteurs)
selected_sensors = ['sensor_7', 'sensor_8', 'sensor_12', 'sensor_15']
plt.figure(figsize=(14, 8))
for sensor in selected_sensors:
    plt.plot(engine_data['time_cycles'], engine_data[sensor], label=sensor)
plt.title(f"Évolution multi-capteurs – Moteur {engine_id}")
plt.legend(); plt.grid(True); plt.show()

# 5. IDENTIFICATION DES CONSTANTS (Pour coordination avec Phase 2)
sensor_columns = [col for col in df.columns if col.startswith("sensor")]
sensor_variance = df[sensor_columns].var()
constant_sensors = sensor_variance[sensor_variance == 0].index.tolist()
print("\n===== Capteurs constants (variance = 0) =====")
print(constant_sensors)

# 6. CORRÉLATION ET BOXPLOTS (Restauré)
plt.figure(figsize=(16, 12))
sns.heatmap(df[sensor_columns].corr(), cmap="coolwarm", annot=False)
plt.title("Matrice de corrélation des capteurs"); plt.show()

plt.figure(figsize=(16, 6))
df[sensor_columns[:10]].boxplot()
plt.title("Boxplot – Premiers capteurs"); plt.xticks(rotation=45); plt.show()