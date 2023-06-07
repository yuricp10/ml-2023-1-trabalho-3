import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.base import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Carregando o dataset
dados = pd.read_csv('ds_salaries.csv')

print("verificando dataset")
print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))

print(dados.head)
print(dados.tail)

print("analisando dados faltantes")
print(dados.isnull().sum())


print("Printando nome das colunas:")
print(dados.columns)

print("Teste de normalizar os valores dentro de 'remote_ratio'")

scaler = MinMaxScaler(feature_range=(0,1))
# dados['remote_ratio_normalizado'] = scaler.fit_transform(dados[['remote_ratio']])
# dados['salary_in_usd_normalizado'] = scaler.fit_transform(dados[['salary_in_usd']])

print("As colunas agora são:", dados.columns)
print("Printando para ver a nova coluna")
print(dados)

features = ['experience_level', 'employment_type', 'job_title', 'remote_ratio', 'company_location', 'company_size', 'salary_currency']
target = 'salary_in_usd'

dados_selecionados = dados[features + [target]]

print(dados_selecionados)

# dados_selecionados = pd.get_dummies(dados_selecionados['company_location'], prefix='company_location')

dados_tratados = dados_selecionados.copy()




company_size_mapeamento = {"L":3, "M":2, "S":1}

experience_level_mapeamento = {"EN":1, "MI":2, "SE":3, "EX":4}


dados_tratados['company_size'] = dados_selecionados['company_size'].map(company_size_mapeamento)

dados_tratados['experience_level'] = dados_selecionados['experience_level'].map(experience_level_mapeamento)

for coluna in dados_tratados.columns:
    if  dados_tratados[coluna].dtype == "object":
        dados_tratados[coluna] = pd.Categorical(dados_tratados[coluna]).codes


print("Dados tratados antes da normalização: \n",dados_tratados.head(20))

dados_tratados = pd.DataFrame(scaler.fit_transform(dados_tratados.values), columns=dados_tratados.columns)


print(dados_tratados.columns)

print(dados_tratados.shape[1])

print("Dados tratados depois da normalização",dados_tratados.head(20))



X = dados_tratados.drop('salary_in_usd',axis = 1)
y = dados_tratados['salary_in_usd']

#Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Criando a mlp
model = Sequential()
model.add(Dense(7,
               activation='relu',
               input_shape=(7,),
               name='input'))
model.add(Dense(64,
                activation='relu',
                name = 'inter01'))
model.add(Dropout(0.2,name='drop01'))
model.add(Dense(64,
                activation='relu',
                name='inter02'))
# model.add(Dropout(0.2, name='drop02'))
model.add(Dense(7,
                activation='relu',
                name='inter03'))

model.add(Dense(1,
                activation='linear',
                name='output'))
model.compile(optimizer=Adam(), 
              loss='mean_squared_error')
model.summary()


print("Verificando shape da x_test", X_test.shape)

#Adicionando o earlystopping e o checkpoint para armazenar o melhor resultado
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 150, min_delta=0.001,restore_best_weights=True)

mc = keras.callbacks.ModelCheckpoint('melhor_modelo.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#Realizando o treinamento (nos testes não consegui achar uma configuração boa para abordar este problema. Meu melhor resultado foi um r^2 = 0.39)
history = model.fit(X_train, y_train, epochs=500, batch_size=36, validation_data=(X_test, y_test),verbose = 1, callbacks=[es, mc])

loss = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")

num_outputs = model.output_shape[-1]
print("Número de saídas do modelo:", num_outputs)

print("X_test {} e x_train {}".format(X_test, X_train))



predictions = model.predict(X_test)

print("Dimensão de predictions", predictions.shape)

# Calcular as métricas
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Imprimir as métricas
print("MAE:", "{:.2f}".format(mae) )
print("RMSE:", "{:.2f}".format(rmse))
print("R^2:", "{:.2f}".format(r2))


# Mapeamento das categorias
categorias_experience_level = dados['experience_level'].astype('category').cat.categories

categorias_company_location = dados['company_location'].astype('category').cat.categories

categorias_employment_type = dados['employment_type'].astype('category').cat.categories

categorias_job_title = dados['job_title'].astype('category').cat.categories

categorias_company_size = dados['company_size'].astype('category').cat.categories

categorias_salary_currency = dados['salary_currency'].astype('category').cat.categories



# Recebendo os inputs para depois realizar a previsao e testar a mlp
print("Informe os valores do profissional para realizar o teste:\n")

experience_level = input("Nível de experiência (EN , MI, SE, EX): ").upper()
employment_type = input("Tipo de emprego (FT, PT, CT, FL): ").upper()
job_title = input("Cargo (Data Engineer, Data Scientist, Data Analyst...): ").title()
remote_ratio = int(input("Taxa de trabalho remoto (0 a 100): "))
company_location = input("Localização da empresa (US, ES, CA...): ").upper()
company_size = input("Tamanho da empresa (L, M, S): ").upper()
salary_currency = input("Moeda do salário (USD, EUR, CHF...): ").upper()



# Obtendo os valores numéricos correspondentes 
valor_numerico_el = categorias_experience_level.get_loc(experience_level)
print("Valor numérico:", valor_numerico_el)

valor_numerico_et = categorias_employment_type.get_loc(employment_type)
print("Valor numérico:", valor_numerico_et)

valor_numerico_jt = categorias_job_title.get_loc(job_title)
print("Valor numérico:", valor_numerico_jt)

valor_numerico_cl = categorias_company_location.get_loc(company_location)
print("Valor numérico:", valor_numerico_cl)

valor_numerico_cz = categorias_company_size.get_loc(company_size)
print("Valor numérico:", valor_numerico_cz)

valor_numerico_sc = categorias_salary_currency.get_loc(salary_currency)
print("Valor numérico:", valor_numerico_sc)

# Alocando os dados recebidos em um array 

dados_usuario = [
    valor_numerico_el,
    valor_numerico_et,
    valor_numerico_jt,
    remote_ratio,
    valor_numerico_cl,
    valor_numerico_cz,
    valor_numerico_sc
]

#Manipulando o shape para que consigam ser feitas as predições e normalizações
dados_usuario = np.array(dados_usuario, ndmin =2)
dados_usuario = dados_usuario.reshape(-1,1)

dados_usuario_norm = scaler.fit_transform(dados_usuario)

dados_usuario_norm = dados_usuario_norm.reshape(1,7)

previsao = model.predict(dados_usuario_norm)

resultado = scaler.inverse_transform(previsao)

#Mostrando os resultados
# print("Salário previsto:", previsao[0])
print("Salário previsto é aproximadamente:", int(resultado[0][0]*1000),"USD")

