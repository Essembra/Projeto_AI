"""
Tema 5 - Deteção de Anomalias e Ameaças de Segurança
Primeira Análise Exploratória de Dados (EDA) focada no desequilíbrio de classes.

Uso:  python eda_desequilibrio.py
Requisitos: pip install pandas openpyxl matplotlib seaborn scikit-learn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------
# 0. Configuração
# ----------------------------------------------------------------------------
FICHEIRO = "Tema05_Deteccao_Anomalias_Seguranca.xlsx"
FOLHA = "Trafego_Rede"
SAIDA = Path("figuras_eda")
SAIDA.mkdir(exist_ok=True)
SEMENTE = 42  # reprodutibilidade
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# 1. Carregar os dados
# ----------------------------------------------------------------------------
df = pd.read_excel(FICHEIRO, sheet_name=FOLHA)  # a folha LEIA-ME é ignorada
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Dimensão: {df.shape[0]} linhas x {df.shape[1]} colunas")

# ----------------------------------------------------------------------------
# 2. Qualidade dos dados
# ----------------------------------------------------------------------------
print("\n--- Tipos e valores em falta ---")
print(pd.DataFrame({"tipo": df.dtypes, "nulos": df.isna().sum()}))
print("\nLinhas duplicadas:", df.duplicated().sum(),
      "| IDs duplicados:", df["id_fluxo"].duplicated().sum())

# ----------------------------------------------------------------------------
# 3. Desequilíbrio de classes (binário e multiclasse)
# ----------------------------------------------------------------------------
cont_bin = df["ataque"].value_counts().sort_index()
print("\n--- Alvo binário ('ataque') ---")
print(pd.DataFrame({"n": cont_bin, "%": (cont_bin / len(df) * 100).round(2)}))

cont_cls = df["classe"].value_counts()
print("\n--- Alvo multiclasse ('classe') ---")
print(pd.DataFrame({"n": cont_cls, "%": (cont_cls / len(df) * 100).round(2)}))

# Rácio de desequilíbrio (IR) = maioria / minoria
ir = cont_cls.max() / cont_cls.min()
print(f"\nRácio de desequilíbrio (normal / classe mais rara): {ir:.1f} : 1")

# Porque é que a exatidão (accuracy) engana: classificador "burro" que diz sempre "normal"
acc_baseline = (df["ataque"] == 0).mean()
print(f"Accuracy de um modelo que prevê SEMPRE 'normal': {acc_baseline:.3f} "
      f"(recall de ataques = 0.0)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=cont_bin.index.map({0: "Normal", 1: "Ataque"}), y=cont_bin.values, ax=ax[0])
ax[0].set_title("Distribuição binária")
ax[0].set_ylabel("Nº de fluxos")
sns.barplot(x=cont_cls.index, y=cont_cls.values, ax=ax[1], order=cont_cls.index)
ax[1].set_yscale("log")  # escala log para ver as classes raras
ax[1].set_title("Distribuição multiclasse (escala log)")
ax[1].tick_params(axis="x", rotation=30)
for a in ax:
    for c in a.containers:
        a.bar_label(c)
plt.tight_layout()
plt.savefig(SAIDA / "01_distribuicao_classes.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 4. Estatísticas por classe (que variáveis separam os ataques?)
# ----------------------------------------------------------------------------
numericas = [
    "duracao_s", "pacotes_enviados", "pacotes_recebidos", "bytes_enviados",
    "bytes_recebidos", "taxa_erro_syn", "ligacoes_mesmo_host_2s",
    "taxa_servicos_diferentes", "portas_destino_distintas_60s",
    "tentativas_login_falhadas", "hora_do_dia",
]
print("\n--- Mediana das variáveis por classe ---")
print(df.groupby("classe")[numericas].median().round(3).T)

# Boxplots (escala log onde faz sentido) - uma figura com todas as variáveis
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for a, col in zip(axes.ravel(), numericas):
    sns.boxplot(data=df, x="classe", y=col, ax=a, order=cont_cls.index)
    if df[col].max() / max(df[col][df[col] > 0].min(), 1e-9) > 1000:
        a.set_yscale("symlog")
    a.set_title(col)
    a.set_xlabel("")
    a.tick_params(axis="x", rotation=40)
for a in axes.ravel()[len(numericas):]:
    a.axis("off")
plt.tight_layout()
plt.savefig(SAIDA / "02_variaveis_por_classe.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------------
# 5. Variáveis categóricas vs ataque
# ----------------------------------------------------------------------------
for col in ["protocolo", "servico", "flag_ligacao"]:
    tab = pd.crosstab(df[col], df["classe"])
    print(f"\n--- {col} x classe (contagens) ---")
    print(tab)
    taxa = df.groupby(col)["ataque"].mean().sort_values(ascending=False)
    print(f"Taxa de ataque por {col}:\n{taxa.round(3)}")

# ----------------------------------------------------------------------------
# 6. Dimensão temporal
# ----------------------------------------------------------------------------
tab_hora = df.groupby("hora_do_dia")["ataque"].agg(["mean", "sum"])
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=tab_hora.index, y=tab_hora["mean"], ax=ax)
ax.set_title("Taxa de ataque por hora do dia")
ax.set_ylabel("Proporção de ataques")
plt.tight_layout()
plt.savefig(SAIDA / "03_ataques_por_hora.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 7. Correlações (só variáveis numéricas + alvo)
# ----------------------------------------------------------------------------
corr = df[numericas + ["ataque"]].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlação de Spearman")
plt.tight_layout()
plt.savefig(SAIDA / "04_correlacoes.png", dpi=150)
plt.close()
print("\n--- Correlação de cada variável com 'ataque' ---")
print(corr["ataque"].drop("ataque").sort_values(key=abs, ascending=False).round(3))

# ----------------------------------------------------------------------------
# 8. Casos difíceis ("normal" com aspeto de ataque) -> origem dos falsos positivos
# ----------------------------------------------------------------------------
normal = df[df["classe"] == "normal"]
# Comparar com o que se vê em cada ataque: PortScan tem 47-886 portas; Exfiltração tem volumes enormes
scanner_like = normal[normal["portas_destino_distintas_60s"] >= 30]
limiar_bytes = 10_000_000  # 10 MB: volume extremo (ver boxplot de bytes_enviados)
backup_like = normal[normal["bytes_enviados"] >= limiar_bytes]
print(f"\nNormais com >=30 portas distintas (parecem PortScan; scanners autorizados): {len(scanner_like)}")
print(f"Normais com bytes enviados >= 10 MB (parecem Exfiltração; backups): {len(backup_like)}")
print(backup_like[["servico", "bytes_enviados", "hora_do_dia"]].head())

# ----------------------------------------------------------------------------
# 9. Verificações antes de modelar (evitar fuga de informação / data leakage)
# ----------------------------------------------------------------------------
print("\n--- Notas para a modelação ---")
print("* Excluir 'id_fluxo', 'timestamp' e 'ip_origem' das features (identificadores).")
print("* 'classe' e 'ataque' são alvos: nunca usar um como feature do outro.")
print("* Fazer o split estratificado ANTES de SMOTE/normalização (aplicar só no treino).")

from sklearn.model_selection import train_test_split

X = df.drop(columns=["id_fluxo", "timestamp", "ip_origem", "classe", "ataque"])
y = df["ataque"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMENTE
)
print(f"\nTreino: {len(y_tr)} (ataques {y_tr.mean():.2%}) | "
      f"Teste: {len(y_te)} (ataques {y_te.mean():.2%})")
print(f"\nFiguras guardadas em: {SAIDA.resolve()}")