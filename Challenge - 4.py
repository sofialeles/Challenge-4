import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Criação do DataFrame de Amostra
# Em um cenário real, aqui você carregaria seus dados (ex: pd.read_csv('seu_arquivo.csv'))
data = {'gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'Partner': ['Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No'],
        'tenure': [1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 58, 49],
        'PhoneService': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'No', 'No phone service', 'No', 'Yes', 'Yes', 'No phone service', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
        'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
        'StreamingTV': ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
        'StreamingMovies': ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check', 'Electronic check', 'Credit card (automatic)', 'Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Electronic check', 'Credit card (automatic)', 'Electronic check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 56.15, 99.85, 20.15, 103.70, 69.20],
        'TotalCharges': [29.85, 1889.50, 108.15, 1840.75, 151.65, 820.5, 1949.4, 301.9, 3046.05, 3487.95, 1312.4, 334.3, 5979.4, 3357.9],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No']}
df = pd.DataFrame(data)

# 2. Pré-processamento dos Dados
# Converte 'TotalCharges' para numérico e preenche valores nulos com a mediana
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Transforma variáveis categóricas em numéricas (one-hot encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Definição das Variáveis e Divisão do Conjunto de Dados
# Define a variável alvo (y) e as variáveis preditoras (X)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinamento do Modelo de Machine Learning
# Inicializa e treina o classificador de Árvore de Decisão
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 5. Avaliação do Modelo
# Faz previsões no conjunto de teste
y_pred = dt_classifier.predict(X_test)

# Calcula e imprime a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {accuracy:.2f}\n")

# Imprime o relatório de classificação detalhado
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Calcula e imprime a matriz de confusão
print("Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 6. Análise de Importância das Variáveis
# Extrai a importância de cada variável
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
})
# Ordena as variáveis pela importância e seleciona as 10 principais
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

# 7. Geração dos Gráficos
# Gráfico de Importância das Variáveis
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('Top 10 Fatores Mais Importantes para o Churn')
plt.xlabel('Nível de Importância')
plt.ylabel('Fator')
plt.tight_layout()
# Salva a imagem em um arquivo
# plt.savefig('feature_importance.png')
plt.show()


# Gráfico da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Deu Churn', 'Deu Churn'], 
            yticklabels=['Não Deu Churn', 'Deu Churn'])
plt.ylabel('Valor Real')
plt.xlabel('Previsão do Modelo')
plt.title('Matriz de Confusão')
plt.tight_layout()
# Salva a imagem em um arquivo
# plt.savefig('confusion_matrix.png')
plt.show()