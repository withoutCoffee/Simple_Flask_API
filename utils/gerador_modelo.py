import numpy as np 
import matplotlib.pyplot as plt 

## ML
from sklearn.linear_model import LinearRegression
import pickle

b = np.random.randint(20)

np.random.seed(7)

x,y = [],[]
for i in range(60):
    x.append(i + np.random.normal(0,6,1) +b)
    y.append(i + np.random.normal(0,3,1) + b*2)

# Iniciando e treinando o modelo

model = LinearRegression()
model.fit(x,y)

# Analizando a resposta
print(model.predict([[0],[1000]]))

# Salvando o modelo
with open('model.pickle','wb') as f:
    pickle.dump(model,f)

# Carregando modelo para o código 
with open('model.pickle','rb') as f:

    modelo = pickle.load(f)

print('modelo carregado!')
print(modelo.predict([[0],[1000]]))

# Figure plot
plt.style.use('seaborn-notebook')
plt.scatter(x,y,label='scatter plot com valores x e y')
plt.xlim(0,100)
plt.ylim(0,100)
plt.legend()
plt.title('Gráfico com os valores artificiais gerados')
plt.savefig('grafico.png')

