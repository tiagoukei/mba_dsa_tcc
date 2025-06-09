##### TCC - MBA Data Science e Analytics - USP ESALQ
# Título: Classificação Automatizada de Notas de Manutenção Utilizando Técnicas de PLN em Textos Livres
# Aluno: Tiago Noboru Ukei
# Orientadora: Profa. Dra. Ana Julia Righetto
# Ano: 2025
#####

#%% Importando bibliotecas
import time
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from preprocess import preprocess_text
from collections import Counter
from wordcloud import WordCloud

#%% Importando as tabelas de classificação conforme a ISO 14224
iso_classe_equip = pd.read_excel('../data/Notas de manutenção.xlsx', sheet_name='ISO Classe equipamento')
iso_modo_falha = pd.read_excel('../data/Notas de manutenção.xlsx', sheet_name='ISO Modo falha')

#%% Tratando as tabelas de classificação
iso_classe_equip.rename(columns={'Grp.códs.': 'classe_equip', 'Txt.grp.codific': 'desc_classe_equip'}, inplace=True)
iso_modo_falha.rename(columns={'Codificação': 'modo_falha', 'Txt.code codif.': 'desc_modo_falha'}, inplace=True)

#%% Importando os dados de falhas
notas = pd.read_excel('../data/Notas de manutenção.xlsx', sheet_name='Planilha1')

#%% Tratando os dados de falhas
notas.drop(columns=['Equipamento', 'Tipo de nota', 'Nota', 'Código ABC', 'Txt.grp.codific', 'Txt.code codif.', 'Prioridade', 'Impacto Segurança', 'Impacto Planta', 'Impacto  Func', 'Classif. ocorrência'], inplace=True)
notas.rename(columns={'Data da nota': 'data', 'Descrição': 'desc_nota', 'Denominação do objeto técnico': 'desc_equip', 'Denominação do local de instalação' : 'instalacao', 'Campo ordenação': 'tag_equip', 'Grp.códs.': 'classe_equip', 'Codificação': 'modo_falha'}, inplace=True)
notas['desc_classe_equip'] = notas['classe_equip'].map(iso_classe_equip.set_index('classe_equip')['desc_classe_equip'])
notas['desc_modo_falha'] = notas['modo_falha'].map(iso_modo_falha.set_index('modo_falha')['desc_modo_falha'])

#%% Pré-processamento dos textos
notas['desc_nota_prep'] = notas['desc_nota'].apply(preprocess_text)
notas['desc_equip_prep'] = notas[~notas['desc_equip'].isna()]['desc_equip'].apply(preprocess_text)

#%% Separando as falhas que não possuem classificação
notas_classificar = notas[notas['modo_falha'].isna()]
notas_classificar.reset_index(drop=True, inplace=True)

#%% Separando as falhas que possuem classificação
notas_classificadas = notas[~notas['modo_falha'].isna()]
notas_classificadas.reset_index(drop=True, inplace=True)

#%% Análise descritiva das notas
# Contagem de notas por classe de equipamento
print("Contagem de notas por classe de equipamento:")
classe_equip_count = notas_classificadas['desc_classe_equip'].value_counts()
print(classe_equip_count)

# Contagem de notas por modo de falha
print("\nContagem de notas por modo de falha:")
modo_falha_count = notas_classificadas['desc_modo_falha'].value_counts()
print(modo_falha_count)

#%% Eliminando as notas com menos de 10 ocorrências de classe de equipamento e modo de falha
notas_classificadas = notas_classificadas[notas_classificadas['desc_classe_equip'].map(notas_classificadas['desc_classe_equip'].value_counts()) >= 10]
notas_classificadas = notas_classificadas[notas_classificadas['desc_modo_falha'].map(notas_classificadas['desc_modo_falha'].value_counts()) >= 10]

#%% Análise gráfica das notas
# Contagem de notas por classe de equipamento
plt.figure(figsize=(15, 10))
sns.countplot(data=notas_classificadas, y='desc_classe_equip', order=notas_classificadas['desc_classe_equip'].value_counts().nlargest(20).index)
plt.ylabel('Classe de Equipamento')
plt.xlabel('Contagem de Notas')

# Contagem de notas por modo de falha
plt.figure(figsize=(15, 10))
sns.countplot(data=notas_classificadas, y='desc_modo_falha', order=notas_classificadas['desc_modo_falha'].value_counts().nlargest(20).index)
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas')

# Modos de falha mais frequentes nas principais classes de equipamento
valvulas = notas_classificadas[notas_classificadas['classe_equip'] == 'ISOB9VA']
plt.figure(figsize=(15, 10))
sns.countplot(data=valvulas, y='desc_modo_falha', order=valvulas['desc_modo_falha'].value_counts().index)
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas') 

bombas = notas_classificadas[notas_classificadas['classe_equip'] == 'ISOB6PU']
plt.figure(figsize=(15, 10))
sns.countplot(data=bombas, y='desc_modo_falha', order=bombas['desc_modo_falha'].value_counts().index)
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas') 

#%% Separando as notas que possuem descrição do equipamento
notas_classificadas_equip = notas_classificadas[notas_classificadas['desc_equip_prep'].notna()]
notas_classificadas_equip.reset_index(drop=True, inplace=True)

#%% Concatenando as colunas de descrição do equipamento e da nota
notas_classificadas_equip['desc_equip_nota'] = notas_classificadas_equip['desc_equip_prep'] + ' ' + notas_classificadas_equip['desc_nota_prep']

#%% Contando as palavras mais frequentes nas descrições dos equipamentos
palavras_equip = notas_classificadas_equip['desc_equip_prep'].str.split(expand=True).stack()
frequencia_equip = Counter(palavras_equip)
top20_equip = dict(frequencia_equip.most_common(20))

#%% Gerando nuvem de palavras para as descrições dos equipamentos
wordcloud_equip = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top20_equip)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_equip, interpolation='bilinear')
plt.axis('off')
plt.show()

#%% Contando as palavras mais frequentes nas descrições das notas
palavras_nota = notas_classificadas_equip['desc_nota_prep'].str.split(expand=True).stack()
frequencia_nota = Counter(palavras_nota)
top20_nota = dict(frequencia_nota.most_common(20))

#%% Gerando nuvem de palavras para as descrições das notas
wordcloud_nota = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top20_nota)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_nota, interpolation='bilinear')
plt.axis('off')
plt.show()

#%% Vetorização dos textos - Bag of Words (BoW)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
X_equip_bow = count_vectorizer.fit_transform(notas_classificadas_equip['desc_equip_prep'])
X_nota_bow = count_vectorizer.fit_transform(notas_classificadas_equip['desc_nota_prep'])
X_equip_nota_bow = count_vectorizer.fit_transform(notas_classificadas_equip['desc_equip_nota'])

#%% Vetorização dos textos - TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_equip_tfidf = tfidf_vectorizer.fit_transform(notas_classificadas_equip['desc_equip_prep'])
X_nota_tfidf = tfidf_vectorizer.fit_transform(notas_classificadas_equip['desc_nota_prep'])
X_equip_nota_tfidf = tfidf_vectorizer.fit_transform(notas_classificadas_equip['desc_equip_nota'])

#%% Separando treino e teste
from sklearn.model_selection import train_test_split
y_classe = notas_classificadas_equip['classe_equip']
y_falha = notas_classificadas_equip['modo_falha']
X_equip_tfidf_train, X_equip_tfidf_test, X_nota_tfidf_train, X_nota_tfidf_test, X_equip_nota_tfidf_train, X_equip_nota_tfidf_test, X_equip_bow_train, X_equip_bow_test, X_nota_bow_train, X_nota_bow_test, X_equip_nota_bow_train, X_equip_nota_bow_test, y_classe_train, y_classe_test, y_falha_train, y_falha_test = train_test_split(X_equip_tfidf, X_nota_tfidf, X_equip_nota_tfidf, X_equip_bow, X_nota_bow, X_equip_nota_bow, y_classe, y_falha, test_size=0.2, stratify=y_classe, random_state=42)
dados_classe = [
    ('TF-IDF', X_equip_tfidf_train, y_classe_train, X_equip_tfidf_test, y_classe_test),
    ('BoW', X_equip_bow_train, y_classe_train, X_equip_bow_test, y_classe_test)
]
dados_falha = [
    ('TF-IDF (nota)', X_nota_tfidf_train, y_falha_train, X_nota_tfidf_test, y_falha_test),
    ('TF-IDF (equip + nota)', X_equip_nota_tfidf_train, y_falha_train, X_equip_nota_tfidf_test, y_falha_test),
    ('BoW (nota)', X_nota_bow_train, y_falha_train, X_nota_bow_test, y_falha_test),
    ('BoW (equip + nota)', X_equip_nota_bow_train, y_falha_train, X_equip_nota_bow_test, y_falha_test)
]

#%% Criando os modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

modelos = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    #('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(kernel='linear', random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Naive Bayes', MultinomialNB()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=100, random_state=42))
]

#%% Treinando e comparando os modelos - Classes de Equipamento
comparacao_modelos_classe = pd.DataFrame(columns=['Modelo', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
y_classe = pd.DataFrame()
y_classe['Teste'] = y_classe_test

for vectorizer, X_train, y_train, X_test, y_test in dados_classe:
    print(f"\nTreinando modelos com {vectorizer}...")

    for nome_modelo, modelo in modelos:
        # Treinamento do modelo
        print(f"Treinando {nome_modelo} com {vectorizer}...")
        start_time = time.time()
        modelo.fit(X_train, y_train)
        end_time = time.time()
        
        # Avaliação do modelo
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Acurácia ({nome_modelo} - {vectorizer}):", accuracy)
        print(f"Acurácia balanceada ({nome_modelo} - {vectorizer}):", balanced_accuracy)
        print(f"Precisão ({nome_modelo} - {vectorizer}):", precision)
        print(f"Recall ({nome_modelo} - {vectorizer}):", recall)
        print(f"F1 Score ({nome_modelo} - {vectorizer}):", f1)
        print(f"Tempo de treinamento ({nome_modelo} - {vectorizer}): {end_time - start_time:.2f} segundos")
        print(f"Relatório de Classificação ({nome_modelo} - {vectorizer}):\n", classification_report(y_test, y_pred))
        comparacao_modelos_classe.loc[len(comparacao_modelos_classe)] = [nome_modelo + ' - ' + vectorizer, accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]
        y_classe[nome_modelo + ' - ' + vectorizer] = y_pred

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão - {nome_modelo} - {vectorizer}')
        plt.show()

#%% Treinando e comparando os modelos - Modos de Falha
comparacao_modelos_falha = pd.DataFrame(columns=['Modelo', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
y_falha = pd.DataFrame()
y_falha['Teste'] = y_falha_test

for vectorizer, X_train, y_train, X_test, y_test in dados_falha:
    print(f"\nTreinando modelos com {vectorizer}...")

    for nome_modelo, modelo in modelos:
        # Treinamento do modelo
        print(f"Treinando {nome_modelo} com {vectorizer}...")
        start_time = time.time()
        modelo.fit(X_train, y_train)
        end_time = time.time()
        
        # Avaliação do modelo
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Acurácia ({nome_modelo} - {vectorizer}):", accuracy)
        print(f"Acurácia balanceada ({nome_modelo} - {vectorizer}):", balanced_accuracy)
        print(f"Precisão ({nome_modelo} - {vectorizer}):", precision)
        print(f"Recall ({nome_modelo} - {vectorizer}):", recall)
        print(f"F1 Score ({nome_modelo} - {vectorizer}):", f1)
        print(f"Tempo de treinamento ({nome_modelo} - {vectorizer}): {end_time - start_time:.2f} segundos")
        print(f"Relatório de Classificação ({nome_modelo} - {vectorizer}):\n", classification_report(y_test, y_pred))
        comparacao_modelos_falha.loc[len(comparacao_modelos_falha)] = [nome_modelo + ' - ' + vectorizer, accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]
        y_falha[nome_modelo + ' - ' + vectorizer] = y_pred

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão - {nome_modelo} - {vectorizer}')
        plt.show()
