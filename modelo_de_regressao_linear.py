# autor: lucasabreuzip
# modelo de regressão linear simples com validação cruzada
# usando o dataset aerogerador.dat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # grafico
import seaborn as sns            # grafico
from sklearn.linear_model import LinearRegression  # o modelo de regressão linear
from sklearn.metrics import mean_squared_error, mean_absolute_error# métricas de avaliação
from sklearn.model_selection import train_test_split # separa treino/teste de forma aleatória.
import os

data = pd.read_csv('aerogerador.dat', sep='\t', header=None) # sobe os dados (o \t e porque é separado por tabulação pq nao tem cabeçalho)
data.columns = ['velocidade', 'potencia']

plt.scatter(data['velocidade'], data['potencia'], alpha=0.5)
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potencia Gerada (kW)')
plt.title('Dados Carregado : Velocidade x Potencia')
plt.grid(True)
plt.show()

X = data[['velocidade']].values
y = data['potencia'].values

mse_list = []
mae_list = []

#==========
# CONFIGURACAO
#==========

num_rodadas = 5000  # Defina o número de rodadas
teste_percent = 0.2  # divide os dados em treino (80%) e teste (20%) aleatoriamente
treino_percent = 1 - teste_percent

# Atualizar o loop para usar num_rodadas
for _ in range(num_rodadas): # rodadas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=teste_percent) 
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # armazenado em lista
    mse_list.append(mse)
    mae_list.append(mae)

def resumo_metricas(nome, lista):
    arr = np.array(lista)
    print(f"{nome} - Media: {arr.mean():.4f}, Desvio Padrao: {arr.std():.4f}, Maximo: {arr.max():.4f}, Minimo: {arr.min():.4f}")

print(f"Resultados apos {num_rodadas} rodadas:")
resumo_metricas("MSE", mse_list)
resumo_metricas("MAE", mae_list)

# Visualização das metricas de MSE e MAE usando seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(mse_list, kde=True, ax=axs[0], color='blue')
axs[0].set_title('Distribuição MSE')
axs[0].set_xlabel('MSE')
axs[0].set_ylabel('Frequencia')

sns.histplot(mae_list, kde=True, ax=axs[1], color='green')
axs[1].set_title('Distribuição MAE')
axs[1].set_xlabel('MAE')
axs[1].set_ylabel('Frequencia')

plt.tight_layout()
plt.show()

#visualização
model = LinearRegression()
model.fit(X, y)
data['potencia_predita'] = model.predict(X)


#grafico de comparação entre valores reais e preditos
plt.figure(figsize=(10, 6))
plt.scatter(data['potencia'], data['potencia_predita'], alpha=0.5, color='blue', label='Valores Preditos')
plt.scatter(data['potencia'], data['potencia'], alpha=0.5, color='green', label='Valores Reais')  # Adicionado valores reais em verde
plt.plot([data['potencia'].min(), data['potencia'].max()], [data['potencia'].min(), data['potencia'].max()], color='orange', linestyle='--', label='Linha Ideal')
plt.xlabel('Potencia Real (kW)')
plt.ylabel('Potencia Predita (kW)')
plt.title('Comparação: Valores Reais x Preditos')
plt.legend()
plt.grid(True)
plt.show()

#salvar os resultados pasta
resultados_dir = 'teste_de_resultados_do_modelo'

#criar o diretório de resultados
if not os.path.exists(resultados_dir):
    os.makedirs(resultados_dir)

resultados_existentes = len([f for f in os.listdir(resultados_dir) if f.startswith('resultados_metricas_') and f.endswith('.txt')])
resultados_path = os.path.join(resultados_dir, 'resultados_metricas.txt')

if os.path.exists(resultados_path):
    with open(resultados_path, 'r') as f:
        linhas = f.readlines()
        execucoes_existentes = sum(1 for linha in linhas if linha.startswith('Execucao'))
else:
    execucoes_existentes = 0

numero_execucao = execucoes_existentes + 1

#info que vai para o arquivo .txt
with open(resultados_path, 'a') as f:
    f.write(f"Execucao ({numero_execucao}) - Resultados apos ({num_rodadas}) rodadas com ({treino_percent*100:.0f}%) treino e ({teste_percent*100:.0f}%) teste:\n")
    f.write(f"MSE - Media: {np.mean(mse_list):.4f}, Desvio Padrao: {np.std(mse_list):.4f}, Maximo: {np.max(mse_list):.4f}, Minimo: {np.min(mse_list):.4f}\n")
    f.write(f"MAE - Media: {np.mean(mae_list):.4f}, Desvio Padrao: {np.std(mae_list):.4f}, Maximo: {np.max(mae_list):.4f}, Minimo: {np.min(mae_list):.4f}\n\n")