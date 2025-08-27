##### TCC - MBA Data Science e Analytics - USP ESALQ
# Título: Classificação Automatizada de Notas de Manutenção Utilizando Técnicas de PLN em Textos Livres
# Aluno: Tiago Noboru Ukei
# Orientadora: Profa. Dra. Ana Julia Righetto
# Ano: 2025
#####

#%% Importando bibliotecas
import time
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from preprocess import preprocess_text
from collections import Counter
from wordcloud import WordCloud
from pathlib import Path

#%% Função para aplicar as normas da USP-ESALQ em gráficos
def aplicar_normas(ax, fonte='Arial', tam=11, cor='black', lw=1.5):
    # 1) Sem grade
    ax.grid(False)

    # 2) Borda “inexistente”: some com top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3) Eixos principais (left/bottom) em preto e 1,5 pt
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(cor)
        ax.spines[side].set_linewidth(lw)

    # 4) Títulos dos eixos (Arial, ≤11, preto)
    ax.set_xlabel(ax.get_xlabel(), fontname=fonte, fontsize=tam, color=cor)
    ax.set_ylabel(ax.get_ylabel(), fontname=fonte, fontsize=tam, color=cor)

    # 5) Fundo “sem preenchimento” (branco)
    ax.set_facecolor('white')

#%% Definindo o caminho do arquivo de dados
DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'Notas de manutenção.xlsx'

#%% Arquivo para armazenar os resultados dos modelos
RESULTS_FILE = Path(__file__).resolve().parents[1] / 'results' / 'Notas classificadas.xlsx'
OUTPUT_FILE = Path(__file__).resolve().parents[1] / 'results' / 'output.txt'

#%% Abrindo o arquivo de saída
import sys
sys.stdout = open(OUTPUT_FILE, 'w', encoding='utf-8')

#%% Importando as tabelas de classificação conforme a ISO 14224
iso_classe_equip = pd.read_excel(DATA_FILE, sheet_name='ISO Classe equipamento')
iso_modo_falha = pd.read_excel(DATA_FILE, sheet_name='ISO Modo falha')

#%% Tratando as tabelas de classificação
iso_classe_equip.rename(columns={'Grp.códs.': 'classe_equip', 'Txt.grp.codific': 'desc_classe_equip'}, inplace=True)
iso_modo_falha.rename(columns={'Codificação': 'modo_falha', 'Txt.code codif.': 'desc_modo_falha'}, inplace=True)

#%% Importando os dados de falhas
notas = pd.read_excel(DATA_FILE, sheet_name='Notas')

#%% Organizando as colunas da tabela de notas
notas.drop(columns=['Equipamento', 'Nota', 'Código ABC', 'Txt.grp.codific', 'Txt.code codif.', 'Txt.grp.codific - novo', 'Txt.code codif. - novo', 'Prioridade', 'Impacto Segurança', 'Impacto Planta', 'Impacto  Func', 'Classif. ocorrência'], inplace=True)
notas.rename(columns={'Data da nota': 'data', 'Tipo de nota': 'tipo', 'Descrição': 'desc_nota', 'Denominação do objeto técnico': 'desc_equip', 'Denominação do local de instalação' : 'instalacao', 'Campo ordenação': 'tag_equip', 'Grp.códs.': 'classe_equip', 'Codificação': 'modo_falha', 'Grp.códs. - novo': 'classe_equip_novo', 'Codificação - novo': 'modo_falha_novo'}, inplace=True)
notas['desc_classe_equip'] = notas['classe_equip'].map(iso_classe_equip.set_index('classe_equip')['desc_classe_equip'])
notas['desc_modo_falha'] = notas['modo_falha'].map(iso_modo_falha.set_index('modo_falha')['desc_modo_falha'])
notas['desc_classe_equip_novo'] = notas['classe_equip_novo'].map(iso_classe_equip.set_index('classe_equip')['desc_classe_equip'])
notas['desc_modo_falha_novo'] = notas['modo_falha_novo'].map(iso_modo_falha.set_index('modo_falha')['desc_modo_falha'])

#%% Removendo as notas que não possuem descrição do equipamento ou da nota
notas = notas[notas['desc_nota'].notna() & notas['desc_equip'].notna()]
notas.reset_index(drop=True, inplace=True)

#%% Analisando a quantidade de notas por classe de equipamentos e por tipo
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=notas, y='desc_classe_equip', hue='tipo', order=notas['desc_classe_equip'].value_counts().nlargest(25).index)
plt.xlabel('Quantidade de Notas')
plt.ylabel('Classe de Equipamento')
plt.legend(title='Tipo de Nota', loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

#%% Analisando a quantidade de notas por modo de falha e por tipo
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=notas, y='desc_modo_falha', hue='tipo', order=notas['desc_modo_falha'].value_counts().nlargest(25).index)
plt.xlabel('Quantidade de Notas')
plt.ylabel('Modo de Falha')
plt.legend(title='Tipo de Nota', loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

#%% Pré-processamento dos textos
notas['desc_nota_prep'] = notas['desc_nota'].apply(preprocess_text)
notas['desc_equip_prep'] = notas['desc_equip'].apply(preprocess_text)

#%% Removendo as notas que ficaram com descrição vazia após o pré-processamento
notas = notas[(notas['desc_nota_prep'] != '') & (notas['desc_equip_prep'] != '')]
notas.reset_index(drop=True, inplace=True)

#%% Concatenando as colunas de descrição do equipamento e da nota
notas['desc_equip_nota'] = notas['desc_equip_prep'] + ' ' + notas['desc_nota_prep']

#%% Separando as notas M2
notas_m2 = notas[notas['tipo'] == 'M2'].copy()
notas_m2.reset_index(drop=True, inplace=True)

#%% Análise descritiva das notas
# Contagem de notas por classe de equipamento
contagem_classe_equip = pd.DataFrame(notas_m2[notas_m2['desc_classe_equip'].notna()]['desc_classe_equip'].unique(), columns=['Classe de Equipamento'])
contagem_classe_equip['Original'] = contagem_classe_equip['Classe de Equipamento'].map(notas_m2['desc_classe_equip'].value_counts()).fillna(0).astype(int)
contagem_classe_equip['Reclassificado'] = contagem_classe_equip['Classe de Equipamento'].map(notas_m2['desc_classe_equip_novo'].value_counts()).fillna(0).astype(int)
contagem_classe_equip.sort_values(by='Reclassificado', ascending=False, inplace=True)

print("Contagem de notas por classe de equipamento:")
print(contagem_classe_equip.to_string(index=False))

# Contagem de notas por modo de falha
contagem_modo_falha = pd.DataFrame(notas_m2[notas_m2['desc_modo_falha'].notna()]['desc_modo_falha'].unique(), columns=['Modo de Falha'])
contagem_modo_falha['Original'] = contagem_modo_falha['Modo de Falha'].map(notas_m2['desc_modo_falha'].value_counts()).fillna(0).astype(int)
contagem_modo_falha['Reclassificado'] = contagem_modo_falha['Modo de Falha'].map(notas_m2['desc_modo_falha_novo'].value_counts()).fillna(0).astype(int)
contagem_modo_falha.sort_values(by='Reclassificado', ascending=False, inplace=True)

print("\nContagem de notas por modo de falha:")
print(contagem_modo_falha.to_string(index=False))

#%% Análise gráfica das notas
# Comparando a contagem de notas por classe de equipamento originais e reclassificados no mesmo gráfico lado a lado
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=contagem_classe_equip.head(20).melt(id_vars=['Classe de Equipamento'], value_vars=['Original', 'Reclassificado'], var_name='Classificação', value_name='Contagem de Notas'), y='Classe de Equipamento', x='Contagem de Notas', hue='Classificação', palette=['blue', 'orange'])
plt.ylabel('Classe de Equipamento')
plt.xlabel('Contagem de Notas')
plt.legend(title='Classificação', loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

# Comparando a contagem de notas por modo de falha originais e reclassificados no mesmo gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=contagem_modo_falha.head(20).melt(id_vars=['Modo de Falha'], value_vars=['Original', 'Reclassificado'], var_name='Classificação', value_name='Contagem de Notas'), y='Modo de Falha', x='Contagem de Notas', hue='Classificação', palette=['blue', 'orange'])
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas')
plt.legend(title='Classificação', loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

# Modos de falha mais frequentes nas principais classes de equipamento
valvulas = notas_m2[notas_m2['classe_equip_novo'] == 'ISOB9VA']
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=valvulas, y='desc_modo_falha_novo', order=valvulas['desc_modo_falha_novo'].value_counts().index)
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas') 
aplicar_normas(ax)
plt.tight_layout()

bombas = notas_m2[notas_m2['classe_equip_novo'] == 'ISOB6PU']
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=bombas, y='desc_modo_falha_novo', order=bombas['desc_modo_falha_novo'].value_counts().index)
plt.ylabel('Modo de Falha')
plt.xlabel('Contagem de Notas') 
aplicar_normas(ax)
plt.tight_layout()

#%% Contando as palavras mais frequentes nas descrições dos equipamentos
palavras_equip = notas_m2['desc_equip_prep'].str.split(expand=True).stack()
frequencia_equip = Counter(palavras_equip)
top20_equip = dict(frequencia_equip.most_common(20))

#%% Gerando nuvem de palavras para as descrições dos equipamentos
wordcloud_equip = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top20_equip)
fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(wordcloud_equip, interpolation='bilinear')
plt.axis('off')

#%% Plotando gráfico de barras com as palavras mais comuns nas descrições dos equipamentos
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=list(top20_equip.values()), y=list(top20_equip.keys()), ax=ax)
plt.ylabel('Palavra')
plt.xlabel('Contagem')
aplicar_normas(ax)
plt.tight_layout()

#%% Contando as palavras mais frequentes nas descrições das notas
palavras_nota = notas_m2['desc_nota_prep'].str.split(expand=True).stack()
frequencia_nota = Counter(palavras_nota)
top20_nota = dict(frequencia_nota.most_common(20))

#%% Gerando nuvem de palavras para as descrições das notas
wordcloud_nota = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top20_nota)
fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(wordcloud_nota, interpolation='bilinear')
plt.axis('off')

#%% Plotando gráfico de barras com as palavras mais comuns nas descrições das notas de manutenção
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=list(top20_nota.values()), y=list(top20_nota.keys()), ax=ax)
plt.ylabel('Palavra')
plt.xlabel('Contagem')
aplicar_normas(ax)
plt.tight_layout()

#%% Contando a quantidade de palavras nas descrições dos equipamentos e das notas
notas_m2['n_palavras_equip'] = notas_m2['desc_equip_prep'].str.split().str.len()
notas_m2['n_palavras_nota'] = notas_m2['desc_nota_prep'].str.split().str.len()
notas_m2['n_palavras_equip_nota'] = notas_m2['desc_equip_nota'].str.split().str.len()

#%% Estatísticas descritivas das contagens de palavras
print("Estatísticas descritivas das contagens de palavras nas descrições dos equipamentos:")
print(notas_m2['n_palavras_equip'].describe())
print("\nEstatísticas descritivas das contagens de palavras nas descrições das notas:")
print(notas_m2['n_palavras_nota'].describe())
print("\nEstatísticas descritivas das contagens de palavras nas descrições dos equipamentos e notas concatenadas:")
print(notas_m2['n_palavras_equip_nota'].describe())

#%% Plotando histogramas das contagens de palavras
min_palavras_equip = notas_m2['n_palavras_equip'].min()
max_palavras_equip = notas_m2['n_palavras_equip'].max()
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(notas_m2['n_palavras_equip'], bins=np.arange(min_palavras_equip, max_palavras_equip + 2) - 0.5)
plt.xlabel('Número de Palavras')
plt.ylabel('Quantidade de Notas')
aplicar_normas(ax)
plt.tight_layout()

min_palavras_nota = notas_m2['n_palavras_nota'].min()
max_palavras_nota = notas_m2['n_palavras_nota'].max()
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(notas_m2['n_palavras_nota'], bins=np.arange(min_palavras_nota, max_palavras_nota + 2) - 0.5)
plt.xlabel('Número de Palavras')
plt.ylabel('Quantidade de Notas')
aplicar_normas(ax)
plt.tight_layout()

min_palavras_equip_nota = notas_m2['n_palavras_equip_nota'].min()
max_palavras_equip_nota = notas_m2['n_palavras_equip_nota'].max()
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(notas_m2['n_palavras_equip_nota'], bins=np.arange(min_palavras_equip_nota, max_palavras_equip_nota + 2) - 0.5)
plt.xlabel('Número de Palavras')
plt.ylabel('Quantidade de Notas')
aplicar_normas(ax)
plt.tight_layout()

#%% Número de palavras únicas nas descrições dos equipamentos
n_palavras_unicas_equip = len(set(palavras_equip))
print("Número de palavras únicas nas descrições dos equipamentos:", n_palavras_unicas_equip)

#%% Número de palavras únicas nas descrições das notas
n_palavras_unicas_nota = len(set(palavras_nota))
print("Número de palavras únicas nas descrições das notas:", n_palavras_unicas_nota)

#%% Número de palavras únicas nas descrições dos equipamentos e notas concatenadas
n_palavras_unicas_equip_nota = len(set(notas_m2['desc_equip_nota'].str.split(expand=True).stack()))
print("Número de palavras únicas nas descrições dos equipamentos e notas concatenadas:", n_palavras_unicas_equip_nota)

#%% Eliminando as notas com menos de 10 ocorrências de classe de equipamento e modo de falha
notas_m2 = notas_m2[notas_m2['classe_equip'].map(notas_m2['classe_equip'].value_counts()) >= 5]
notas_m2 = notas_m2[notas_m2['modo_falha'].map(notas_m2['modo_falha'].value_counts()) >= 5]
notas_m2 = notas_m2[notas_m2['classe_equip_novo'].map(notas_m2['classe_equip_novo'].value_counts()) >= 5]
notas_m2 = notas_m2[notas_m2['modo_falha_novo'].map(notas_m2['modo_falha_novo'].value_counts()) >= 5]

#%% Vetorização dos textos - Bag of Words (BoW)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
X_eq_bw = count_vectorizer.fit_transform(notas_m2['desc_equip_prep'])
X_nt_bw = count_vectorizer.fit_transform(notas_m2['desc_nota_prep'])
X_eqnt_bw = count_vectorizer.fit_transform(notas_m2['desc_equip_nota'])

#%% Vetorização dos textos - TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_eq_tf = tfidf_vectorizer.fit_transform(notas_m2['desc_equip_prep'])
X_nt_tf = tfidf_vectorizer.fit_transform(notas_m2['desc_nota_prep'])
X_eqnt_tf = tfidf_vectorizer.fit_transform(notas_m2['desc_equip_nota'])

#%% Separando treino e teste
from sklearn.model_selection import train_test_split

# Originais
yc = notas_m2['classe_equip']
yf = notas_m2['modo_falha']

Xc_eq_tf_train, Xc_eq_tf_test, Xc_eqnt_tf_train, Xc_eqnt_tf_test, Xc_eq_bw_train, Xc_eq_bw_test, Xc_eqnt_bw_train, Xc_eqnt_bw_test, yc_train, yc_test = train_test_split(X_eq_tf, X_eqnt_tf, X_eq_bw, X_eqnt_bw, yc, test_size=0.2, stratify=yc, random_state=42)
Xf_nt_tf_train, Xf_nt_tf_test, Xf_eqnt_tf_train, Xf_eqnt_tf_test, Xf_nt_bw_train, Xf_nt_bw_test, Xf_eqnt_bw_train, Xf_eqnt_bw_test, yf_train, yf_test = train_test_split(X_nt_tf, X_eqnt_tf, X_nt_bw, X_eqnt_bw, yf, test_size=0.2, stratify=yf, random_state=42)

# Reclassificados
yc_n = notas_m2['classe_equip_novo']
yf_n = notas_m2['modo_falha_novo']

Xc_n_eq_tf_train, Xc_n_eq_tf_test, Xc_n_eqnt_tf_train, Xc_n_eqnt_tf_test, Xc_n_eq_bw_train, Xc_n_eq_bw_test, Xc_n_eqnt_bw_train, Xc_n_eqnt_bw_test, yc_n_train, yc_n_test = train_test_split(X_eq_tf, X_eqnt_tf, X_eq_bw, X_eqnt_bw, yc_n, test_size=0.2, stratify=yc_n, random_state=42)
Xf_n_nt_tf_train, Xf_n_nt_tf_test, Xf_n_eqnt_tf_train, Xf_n_eqnt_tf_test, Xf_n_nt_bw_train, Xf_n_nt_bw_test, Xf_n_eqnt_bw_train, Xf_n_eqnt_bw_test, yf_n_train, yf_n_test = train_test_split(X_nt_tf, X_eqnt_tf, X_nt_bw, X_eqnt_bw, yf_n, test_size=0.2, stratify=yf_n, random_state=42)

#%% Organizando os dados para treinamento e teste
dados_classe = [
    ('TF-IDF / equip / original', Xc_eq_tf_train, yc_train, Xc_eq_tf_test, yc_test),
    ('TF-IDF / equip + nota / original', Xc_eqnt_tf_train, yc_train, Xc_eqnt_tf_test, yc_test),
    ('BoW / equip / original', Xc_eq_bw_train, yc_train, Xc_eq_bw_test, yc_test),
    ('BoW / equip + nota / original', Xc_eqnt_bw_train, yc_train, Xc_eqnt_bw_test, yc_test),
    ('TF-IDF / equip / reclassificada', Xc_n_eq_tf_train, yc_n_train, Xc_n_eq_tf_test, yc_n_test),
    ('TF-IDF / equip + nota / reclassificada', Xc_n_eqnt_tf_train, yc_n_train, Xc_n_eqnt_tf_test, yc_n_test),
    ('BoW / equip / reclassificada', Xc_n_eq_bw_train, yc_n_train, Xc_n_eq_bw_test, yc_n_test),
    ('BoW / equip + nota / reclassificada', Xc_n_eqnt_bw_train, yc_n_train, Xc_n_eqnt_bw_test, yc_n_test)
]

dados_falha = [
    ('TF-IDF / nota / original', Xf_nt_tf_train, yf_train, Xf_nt_tf_test, yf_test),
    ('TF-IDF / equip + nota / original', Xf_eqnt_tf_train, yf_train, Xf_eqnt_tf_test, yf_test),
    ('BoW / nota / original', Xf_nt_bw_train, yf_train, Xf_nt_bw_test, yf_test),
    ('BoW / equip + nota / original', Xf_eqnt_bw_train, yf_train, Xf_eqnt_bw_test, yf_test),
    ('TF-IDF / nota / reclassificada', Xf_n_nt_tf_train, yf_n_train, Xf_n_nt_tf_test, yf_n_test),
    ('TF-IDF / equip + nota / reclassificada', Xf_n_eqnt_tf_train, yf_n_train, Xf_n_eqnt_tf_test, yf_n_test),
    ('BoW / nota / reclassificada', Xf_n_nt_bw_train, yf_n_train, Xf_n_nt_bw_test, yf_n_test),
    ('BoW / equip + nota / reclassificada', Xf_n_eqnt_bw_train, yf_n_train, Xf_n_eqnt_bw_test, yf_n_test)
]

#%% Criando os modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

modelos = [
    ('Logistic Regression', LogisticRegression(max_iter=1000,random_state=42, n_jobs=-1)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(kernel='linear', random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Naive Bayes', MultinomialNB()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=100, random_state=42))
]

#%% Treinando e comparando os modelos - Classes de Equipamento
comparacao_modelos_classe = pd.DataFrame(columns=['Modelo', 'Vetorizador / Descrição / Classificação', 'Hiperparâmetros', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
yc = pd.DataFrame()
yc['Teste'] = yc_test

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
        hiperparam = modelo.get_params()
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\nResultados do modelo: {nome_modelo}")
        print("Hiperparâmetros:", hiperparam)
        print(f"Acurácia ({nome_modelo} - {vectorizer}):", accuracy)
        print(f"Acurácia balanceada ({nome_modelo} - {vectorizer}):", balanced_accuracy)
        print(f"Precisão ({nome_modelo} - {vectorizer}):", precision)
        print(f"Recall ({nome_modelo} - {vectorizer}):", recall)
        print(f"F1 Score ({nome_modelo} - {vectorizer}):", f1)
        print(f"Tempo de treinamento ({nome_modelo} - {vectorizer}): {end_time - start_time:.2f} segundos")
        print(f"Relatório de Classificação ({nome_modelo} - {vectorizer}):\n", classification_report(y_test, y_pred))
        comparacao_modelos_classe.loc[len(comparacao_modelos_classe)] = [nome_modelo, vectorizer, 'Originais', accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]
        yc[nome_modelo + ' - ' + vectorizer] = y_pred

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão - {nome_modelo} - {vectorizer}')
        aplicar_normas(ax)
        plt.tight_layout()

#%% Gráfico de comparação dos modelos - Classes de Equipamento
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_classe, x='Modelo', y='Acurácia', hue='Vetorizador / Descrição / Classificação', palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_classe, x='Modelo', y='Tempo', hue='Vetorizador / Descrição / Classificação', palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='upper right')
aplicar_normas(ax)
plt.tight_layout()

#%% Treinando e comparando os modelos - Modos de Falha
comparacao_modelos_falha = pd.DataFrame(columns=['Modelo', 'Vetorizador / Descrição / Classificação', 'Hiperparâmetros', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
yf = pd.DataFrame()
yf['Teste'] = yf_test

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
        hiperparam = modelo.get_params()
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\nResultados para {nome_modelo} com {vectorizer}:")
        print("Hiperparâmetros:", hiperparam)
        print(f"Acurácia ({nome_modelo} - {vectorizer}):", accuracy)
        print(f"Acurácia balanceada ({nome_modelo} - {vectorizer}):", balanced_accuracy)
        print(f"Precisão ({nome_modelo} - {vectorizer}):", precision)
        print(f"Recall ({nome_modelo} - {vectorizer}):", recall)
        print(f"F1 Score ({nome_modelo} - {vectorizer}):", f1)
        print(f"Tempo de treinamento ({nome_modelo} - {vectorizer}): {end_time - start_time:.2f} segundos")
        print(f"Relatório de Classificação ({nome_modelo} - {vectorizer}):\n", classification_report(y_test, y_pred))
        comparacao_modelos_falha.loc[len(comparacao_modelos_falha)] = [nome_modelo, vectorizer, 'Originais', accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]
        yf[nome_modelo + ' - ' + vectorizer] = y_pred

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão - {nome_modelo} - {vectorizer}')
        aplicar_normas(ax)
        plt.tight_layout()

#%% Gráfico de comparação dos modelos - Modos de Falha
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_falha, x='Modelo', y='Acurácia', hue='Vetorizador / Descrição / Classificação', palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_falha, x='Modelo', y='Tempo', hue='Vetorizador / Descrição / Classificação', palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='upper right')
aplicar_normas(ax)
plt.tight_layout()

#%% Refinando os modelos - Logistic Regression
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_lr = {
        'C': [1, 10, 100],
        'solver': ['lbfgs', 'sag', 'newton-cg'],
        'max_iter': [1000, 2000]
}

lr_clf = GridSearchCV(estimator=LogisticRegression(random_state=42, n_jobs=-1),
                   param_grid=param_grid_lr,
                   cv=5,
                   scoring='accuracy',
                   verbose=0)

lr_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados:", lr_clf.best_params_)
print("Melhor acurácia:", lr_clf.best_score_)
print("Acurácia no teste:", lr_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_lr = lr_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_lr))
lr_model = LogisticRegression(C=lr_clf.best_params_['C'],
                              solver=lr_clf.best_params_['solver'],
                              max_iter=lr_clf.best_params_['max_iter'],
                              random_state=42)

#%% Refinando os modelos - Random Forest
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_rf = {
    'n_estimators': [100, 200],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
                      param_grid=param_grid_rf,
                      cv=5,
                      scoring='accuracy',
                      verbose=0)

rf_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados para Random Forest:", rf_clf.best_params_)
print("Melhor acurácia:", rf_clf.best_score_)
print("Acurácia no teste:", rf_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_rf = rf_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_rf))
rfModel = RandomForestClassifier(n_estimators=rf_clf.best_params_['n_estimators'], 
                                 criterion=rf_clf.best_params_['criterion'],
                                 max_features=rf_clf.best_params_['max_features'],
                                 max_depth=rf_clf.best_params_['max_depth'], 
                                 min_samples_split=rf_clf.best_params_['min_samples_split'], 
                                 random_state=42)

#%% Refinando os modelos - Gradient Boosting
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

gb_clf = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                      param_grid=param_grid_gb,
                      cv=5,
                      scoring='accuracy',
                      verbose=0)

gb_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados para Gradient Boosting:", gb_clf.best_params_)
print("Melhor acurácia:", gb_clf.best_score_)
print("Acurácia no teste:", gb_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_gb = gb_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_gb))
gbModel = GradientBoostingClassifier(n_estimators=gb_clf.best_params_['n_estimators'], 
                                     learning_rate=gb_clf.best_params_['learning_rate'],
                                     max_depth=gb_clf.best_params_['max_depth'],
                                     min_samples_split=gb_clf.best_params_['min_samples_split'],
                                     random_state=42)

#%% Refinando os modelos - SVM
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm_clf = GridSearchCV(estimator=SVC(random_state=42),
                      param_grid=param_grid_svm,
                      cv=5,
                      scoring='accuracy',
                      verbose=0)

svm_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados para SVM:", svm_clf.best_params_)
print("Melhor acurácia:", svm_clf.best_score_)
print("Acurácia no teste:", svm_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_svm = svm_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_svm))
svmModel = SVC(C=svm_clf.best_params_['C'],
               kernel=svm_clf.best_params_['kernel'],
               gamma=svm_clf.best_params_['gamma'],
               degree=svm_clf.best_params_['degree'],
               random_state=42)

#%% Refinando os modelos - KNN
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'leaf_size': [30, 40, 50]
}

knn_clf = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid_knn,
                      cv=5,
                      scoring='accuracy',
                      verbose=0)

knn_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados para KNN:", knn_clf.best_params_)
print("Melhor acurácia:", knn_clf.best_score_)
print("Acurácia no teste:", knn_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_knn = knn_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_knn))
knnModel = KNeighborsClassifier(n_neighbors=knn_clf.best_params_['n_neighbors'],
                                weights=knn_clf.best_params_['weights'],
                                leaf_size=knn_clf.best_params_['leaf_size'])

#%% Refinando os modelos - Naive Bayes
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
nbModel = MultinomialNB()
nbModel.fit(Xf_n_nt_bw_train, yf_n_train)
y_pred_nb = nbModel.predict(Xf_n_nt_bw_test)
print("Acurácia do Naive Bayes:", accuracy_score(yf_n_test, y_pred_nb))
print("Relatório de Classificação do Naive Bayes:\n", classification_report(yf_n_test, y_pred_nb))

#%% Refinando os modelos - MLP
# Focando nos melhores dados de treino: Vetorização BoW / descrição da nota / notas reclassificadas
param_grid_mlp = {
    'hidden_layer_sizes': [(20, 10), (30, 15)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}
mlp_clf = GridSearchCV(estimator=MLPClassifier(max_iter=1000, random_state=42),
                      param_grid=param_grid_mlp,
                      cv=5,
                      scoring='accuracy',
                      verbose=0)
mlp_clf.fit(Xf_n_nt_bw_train, yf_n_train)
print("Melhores parâmetros encontrados para MLP:", mlp_clf.best_params_)
print("Melhor acurácia:", mlp_clf.best_score_)
print("Acurácia no teste:", mlp_clf.score(Xf_n_nt_bw_test, yf_n_test))
y_pred_mlp = mlp_clf.predict(Xf_n_nt_bw_test)
print("Relatório de Classificação:\n", classification_report(yf_n_test, y_pred_mlp))
mlpModel = MLPClassifier(hidden_layer_sizes=mlp_clf.best_params_['hidden_layer_sizes'],
                         activation=mlp_clf.best_params_['activation'],
                         solver=mlp_clf.best_params_['solver'],
                         alpha=mlp_clf.best_params_['alpha'],
                         learning_rate=mlp_clf.best_params_['learning_rate'],
                         max_iter=1000,
                         random_state=42)

#%% Organizando os modelos refinados
modelos_refinados = [
    ('Logistic Regression', lr_model),
    ('Random Forest', rfModel),
    ('Gradient Boosting', gbModel),
    ('SVM', svmModel),
    ('KNN', knnModel),
    ('Naive Bayes', nbModel),
    ('MLP', mlpModel)
]

#%% Treinando e avaliando os modelos refinados - Classes de Equipamento
comparacao_modelos_classe_refinados = pd.DataFrame(columns=['Modelo', 'Hiperparâmetros', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
for nome_modelo, modelo in modelos_refinados:
    start_time = time.time()
    modelo.fit(Xc_n_eqnt_bw_train, yc_n_train)
    end_time = time.time()
    
    y_pred = modelo.predict(Xc_n_eqnt_bw_test)
    hiperparam = modelo.get_params()
    accuracy = accuracy_score(yc_n_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(yc_n_test, y_pred)
    precision = precision_score(yc_n_test, y_pred, average='macro')
    recall = recall_score(yc_n_test, y_pred, average='macro')
    f1 = f1_score(yc_n_test, y_pred, average='macro')
    print(f"\nResultados do modelo refinado: {nome_modelo}")
    print("Hiperparâmetros:", hiperparam)
    print(f"Acurácia ({nome_modelo}):", accuracy)
    print(f"Acurácia balanceada ({nome_modelo}):", balanced_accuracy)
    print(f"Precisão ({nome_modelo}):", precision)
    print(f"Recall ({nome_modelo}):", recall)
    print(f"F1 Score ({nome_modelo}):", f1)
    print(f"Tempo de treinamento ({nome_modelo}): {end_time - start_time:.2f} segundos")
    print(f"Relatório de Classificação ({nome_modelo}):\n", classification_report(yc_n_test, y_pred))
    comparacao_modelos_classe_refinados.loc[len(comparacao_modelos_classe_refinados)] = [nome_modelo, 'GridSearch', accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]

    cm = confusion_matrix(yc_n_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    aplicar_normas(ax)
    plt.tight_layout()

#%% Gráfico de comparação dos modelos refinados
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_classe_refinados, x='Modelo', y='Acurácia')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_classe_refinados, x='Modelo', y='Tempo')
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Gráfico de comparação do modelo refinado com o modelo inicial com mesmo vetorizador - Classes de Equipamento
comparacao_modelos_classe_refinados= pd.concat([comparacao_modelos_classe_refinados, comparacao_modelos_classe[comparacao_modelos_classe['Vetorizador / Descrição / Classificação'] == 'BoW / equip + nota / reclassificada'].drop(columns=['Vetorizador / Descrição / Classificação'])], ignore_index=True, sort=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_classe_refinados, x='Modelo', y='Acurácia', hue='Hiperparâmetros', hue_order=['Originais', 'GridSearch'], palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

#%% Treinando e avaliando os modelos refinados - Modos de Falha
comparacao_modelos_falha_refinados = pd.DataFrame(columns=['Modelo', 'Hiperparâmetros', 'Acurácia', 'Acurácia Balanceada', 'Precisão', 'Recall', 'F1 Score', 'Tempo'])
for nome_modelo, modelo in modelos_refinados:
    start_time = time.time()
    modelo.fit(Xf_n_nt_bw_train, yf_n_train)
    end_time = time.time()
    
    y_pred = modelo.predict(Xf_n_nt_bw_test)
    hiperparam = modelo.get_params()
    accuracy = accuracy_score(yf_n_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(yf_n_test, y_pred)
    precision = precision_score(yf_n_test, y_pred, average='macro')
    recall = recall_score(yf_n_test, y_pred, average='macro')
    f1 = f1_score(yf_n_test, y_pred, average='macro')
    print(f"\nResultados do modelo refinado: {nome_modelo}")
    print("Hiperparâmetros:", hiperparam)
    print(f"Acurácia ({nome_modelo}):", accuracy)
    print(f"Acurácia balanceada ({nome_modelo}):", balanced_accuracy)
    print(f"Precisão ({nome_modelo}):", precision)
    print(f"Recall ({nome_modelo}):", recall)
    print(f"F1 Score ({nome_modelo}):", f1)
    print(f"Tempo de treinamento ({nome_modelo}): {end_time - start_time:.2f} segundos")
    print(f"Relatório de Classificação ({nome_modelo}):\n", classification_report(yf_n_test, y_pred))
    comparacao_modelos_falha_refinados.loc[len(comparacao_modelos_falha_refinados)] = [nome_modelo, 'GridSearch', accuracy, balanced_accuracy, precision, recall, f1, end_time - start_time]

    cm = confusion_matrix(yf_n_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    aplicar_normas(ax)
    plt.tight_layout()

#%% Gráfico de comparação dos modelos refinados
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_falha_refinados, x='Modelo', y='Acurácia')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_falha_refinados, x='Modelo', y='Tempo')
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Gráfico de comparação do modelo refinado com o modelo inicial com mesmo vetorizador - Modos de Falha
comparacao_modelos_falha_refinados= pd.concat([comparacao_modelos_falha_refinados, comparacao_modelos_falha[comparacao_modelos_falha['Vetorizador / Descrição / Classificação'] == 'BoW / nota / reclassificada'].drop(columns=['Vetorizador / Descrição / Classificação'])], ignore_index=True, sort=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparacao_modelos_falha_refinados, x='Modelo', y='Acurácia', hue='Hiperparâmetros', hue_order=['Originais', 'GridSearch'], palette='Blues')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
plt.legend(loc='lower right')
aplicar_normas(ax)
plt.tight_layout()

#%% Criando modelo roBERTa para classificação de textos em português
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
import torch

# 1) Dados
df = notas_m2[['desc_equip_nota', 'desc_nota_prep', 'classe_equip_novo', 'modo_falha_novo']].dropna().copy()
df['desc_equip_nota'] = df['desc_equip_nota'].astype(str)
df['desc_nota_prep'] = df['desc_nota_prep'].astype(str)

# 2) Labels → ids
le_c = LabelEncoder()
yc_le = le_c.fit_transform(df['classe_equip_novo'])
le_f = LabelEncoder()
yf_le = le_f.fit_transform(df['modo_falha_novo'])

# 3) Split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    df['desc_equip_nota'].tolist(), yc_le, test_size=0.2, random_state=42, stratify=yc_le
)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    df['desc_nota_prep'].tolist(), yf_le, test_size=0.2, random_state=42, stratify=yf_le
)

# 4) Métrica
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w
    }

# 5) Tokenizer/model
model_name = "rdenadai/BR_BERTo"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model_c = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(le_c.classes_))
model_f = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(le_f.classes_))

# 6) Tokenização
enc_c_train = tokenizer(Xc_train, padding=True, return_tensors='pt')
enc_c_test  = tokenizer(Xc_test,  padding=True, return_tensors='pt')
enc_f_train = tokenizer(Xf_train, padding=True, return_tensors='pt')
enc_f_test  = tokenizer(Xf_test,  padding=True, return_tensors='pt')

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_c_ds = HFDataset(enc_c_train, yc_train)
test_c_ds  = HFDataset(enc_c_test,  yc_test)
train_f_ds = HFDataset(enc_f_train, yf_train)
test_f_ds  = HFDataset(enc_f_test,  yf_test)

# 7) Treino
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    logging_dir='./logs',
    dataloader_pin_memory=False,
)

#%% Treinamento dos modelos roBERTa - Classes de Equipamento
trainer_c = Trainer(
    model=model_c,
    args=training_args,
    train_dataset=train_c_ds,
    eval_dataset=test_c_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

start_time = time.time()
trainer_c.train()
end_time = time.time()
roberta_c_time = end_time - start_time
print(f"Tempo total de treinamento do modelo roBERTa - Classes de Equipamento: {roberta_c_time:.2f} segundos")

#%% Treinamento dos modelos roBERTa - Modos de Falha
trainer_f = Trainer(
    model=model_f,
    args=training_args,
    train_dataset=train_f_ds,
    eval_dataset=test_f_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

start_time = time.time()
trainer_f.train()
end_time = time.time()
roberta_f_time = end_time - start_time
print(f"Tempo total de treinamento do modelo roBERTa - Modos de Falha: {roberta_f_time:.2f} segundos")

#%% Avaliação do modelo roBERTa - Classes de Equipamento
print("\n=== Avaliação do modelo roBERTa - Classes de Equipamento ===")
eval_c_metrics = trainer_c.evaluate(test_c_ds)
print("\n=== Métricas finais (teste) ===")
for k, v in eval_c_metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

pred = trainer_c.predict(test_c_ds)
yc_pred = np.argmax(pred.predictions, axis=1)

print("\n=== Classification Report ===")
print(classification_report(yc_test, yc_pred, target_names=le_c.classes_, zero_division=0))

cm = confusion_matrix(yc_test, yc_pred)
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=le_c.classes_, yticklabels=le_c.classes_)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - roBERTa')
aplicar_normas(ax)
plt.tight_layout()

#%% Avaliação do modelo roBERTa - Modos de Falha
print("\n=== Avaliação do modelo roBERTa - Modos de Falha ===")
eval_f_metrics = trainer_f.evaluate(test_f_ds)
print("\n=== Métricas finais (teste) ===")
for k, v in eval_f_metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

pred = trainer_f.predict(test_f_ds)
yf_pred = np.argmax(pred.predictions, axis=1)

print("\n=== Classification Report ===")
print(classification_report(yf_test, yf_pred, target_names=le_f.classes_, zero_division=0))

cm = confusion_matrix(yf_test, yf_pred)
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', norm=LogNorm(), xticklabels=le_f.classes_, yticklabels=le_f.classes_)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - roBERTa')
aplicar_normas(ax)
plt.tight_layout()

#%% Comparando as acurácias dos modelos - Classes de Equipamento
modelos_classe_acuracia = {
    'Logistic Regression': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Logistic Regression', 'Acurácia'].values[0],
    'Random Forest': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Random Forest', 'Acurácia'].values[0],
    'Gradient Boosting': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Gradient Boosting', 'Acurácia'].values[0] if 'Gradient Boosting' in comparacao_modelos_classe_refinados['Modelo'].values else None,
    'SVM': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'SVM', 'Acurácia'].values[0],
    'KNN': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'KNN', 'Acurácia'].values[0],
    'Naive Bayes': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Naive Bayes', 'Acurácia'].values[0],
    'MLP': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'MLP', 'Acurácia'].values[0],
    'roBERTa': eval_c_metrics['eval_accuracy']
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(modelos_classe_acuracia.keys()),
            y=list(modelos_classe_acuracia.values()), ax=ax)
for container in ax.containers:
    ax.bar_label(container,
                 labels=[f'{bar.get_height():.1%}' for bar in container],
                 padding=3)
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Comparando os tempos de treinamento dos modelos - Classes de Equipamento
modelos_classe_tempo = {
    'Logistic Regression': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Logistic Regression', 'Tempo'].values[0],
    'Random Forest': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Random Forest', 'Tempo'].values[0],
    'Gradient Boosting': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Gradient Boosting', 'Tempo'].values[0] if 'Gradient Boosting' in comparacao_modelos_classe_refinados['Modelo'].values else None,
    'SVM': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'SVM', 'Tempo'].values[0],
    'KNN': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'KNN', 'Tempo'].values[0],
    'Naive Bayes': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'Naive Bayes', 'Tempo'].values[0],
    'MLP': comparacao_modelos_classe_refinados.loc[comparacao_modelos_classe_refinados['Modelo'] == 'MLP', 'Tempo'].values[0],
    'roBERTa': roberta_c_time
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(modelos_classe_tempo.keys()),
            y=list(modelos_classe_tempo.values()), ax=ax)
for container in ax.containers:
    ax.bar_label(container,
                 labels=[f'{bar.get_height():.1f}' for bar in container],
                 padding=3)
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.yscale('log')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Comparando as acurácias dos modelos - Modos de Falha
modelos_falha_acuracia = {
    'Logistic Regression': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Logistic Regression', 'Acurácia'].values[0],
    'Random Forest': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Random Forest', 'Acurácia'].values[0],
    'Gradient Boosting': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Gradient Boosting', 'Acurácia'].values[0] if 'Gradient Boosting' in comparacao_modelos_falha_refinados['Modelo'].values else None,
    'SVM': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'SVM', 'Acurácia'].values[0],
    'KNN': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'KNN', 'Acurácia'].values[0],
    'Naive Bayes': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Naive Bayes', 'Acurácia'].values[0],
    'MLP': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'MLP', 'Acurácia'].values[0],
    'roBERTa': eval_f_metrics['eval_accuracy']
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(modelos_falha_acuracia.keys()),
            y=list(modelos_falha_acuracia.values()), ax=ax)
for container in ax.containers:
    ax.bar_label(container,
                 labels=[f'{bar.get_height():.1%}' for bar in container],
                 padding=3)
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Comparando os tempos de treinamento dos modelos - Modos de Falha
modelos_falha_tempo = {
    'Logistic Regression': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Logistic Regression', 'Tempo'].values[0],
    'Random Forest': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Random Forest', 'Tempo'].values[0],
    'Gradient Boosting': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Gradient Boosting', 'Tempo'].values[0] if 'Gradient Boosting' in comparacao_modelos_falha_refinados['Modelo'].values else None,
    'SVM': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'SVM', 'Tempo'].values[0],
    'KNN': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'KNN', 'Tempo'].values[0],
    'Naive Bayes': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'Naive Bayes', 'Tempo'].values[0],
    'MLP': comparacao_modelos_falha_refinados.loc[comparacao_modelos_falha_refinados['Modelo'] == 'MLP', 'Tempo'].values[0],
    'roBERTa': roberta_f_time
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(modelos_falha_tempo.keys()),
            y=list(modelos_falha_tempo.values()), ax=ax)
for container in ax.containers:
    ax.bar_label(container,
                 labels=[f'{bar.get_height():.1f}' for bar in container],
                 padding=3)
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.yscale('log')
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
aplicar_normas(ax)
plt.tight_layout()

#%% Contando a quantidade de classes de equipamento e modos de falha
n_classes_equip = notas_m2['classe_equip_novo'].nunique()
n_modos_falha = notas_m2['modo_falha_novo'].nunique()
print("Quantidade de classes de equipamento:", n_classes_equip)
print("Quantidade de modos de falha:", n_modos_falha)

#%% Clusterização dos textos - Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hierarchical_equip_nota = AgglomerativeClustering(n_clusters=n_classes_equip, metric='euclidean', linkage='ward').fit(X_eqnt_bw.toarray())
hierarchical_nota = AgglomerativeClustering(n_clusters=n_modos_falha, metric='euclidean', linkage='ward').fit(X_nt_bw.toarray())

#%% Adicionando os clusters Hierarchical às notas classificadas
notas_m2['hierarchical_equip_nota'] = hierarchical_equip_nota.labels_
notas_m2['hierarchical_equip_nota'] = notas_m2['hierarchical_equip_nota'].astype('category')
notas_m2['hierarchical_nota'] = hierarchical_nota.labels_
notas_m2['hierarchical_nota'] = notas_m2['hierarchical_nota'].astype('category')

#%% Análise dos clusters - Classes de Equipamentos - Hierarchical
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
from collections import defaultdict
hierarchical_equip_scores = defaultdict(list)
hierarchical_equip_scores['Homogeneity'].append(homogeneity_score(notas_m2['classe_equip'], notas_m2['hierarchical_equip_nota']))
hierarchical_equip_scores['Completeness'].append(completeness_score(notas_m2['classe_equip'], notas_m2['hierarchical_equip_nota']))
hierarchical_equip_scores['V-measure'].append(v_measure_score(notas_m2['classe_equip'], notas_m2['hierarchical_equip_nota']))
hierarchical_equip_scores['Silhouette Score'].append(silhouette_score(X_eqnt_bw, notas_m2['hierarchical_equip_nota'], metric='euclidean'))
print("Hierarchical Clustering - Classes de Equipamentos:")
print(hierarchical_equip_scores)

#%% Análise dos clusters - Modos de Falha - Hierarchical
hierarchical_falha_scores = defaultdict(list)
hierarchical_falha_scores['Homogeneity'].append(homogeneity_score(notas_m2['modo_falha'], notas_m2['hierarchical_nota']))
hierarchical_falha_scores['Completeness'].append(completeness_score(notas_m2['modo_falha'], notas_m2['hierarchical_nota']))
hierarchical_falha_scores['V-measure'].append(v_measure_score(notas_m2['modo_falha'], notas_m2['hierarchical_nota']))
hierarchical_falha_scores['Silhouette Score'].append(silhouette_score(X_nt_bw, notas_m2['hierarchical_nota'], metric='euclidean'))
print("Hierarchical Clustering - Modos de Falha:")
print(hierarchical_falha_scores)

#%% Clusterização dos textos - KMeans
from sklearn.cluster import KMeans
kmeans_equip_nota = KMeans(n_clusters=n_classes_equip, n_init=20).fit(X_eqnt_bw)
kmeans_nota = KMeans(n_clusters=n_modos_falha, n_init=20).fit(X_nt_bw)

#%% Adicionando os clusters KMeans às notas classificadas
notas_m2['kmeans_equip_nota'] = kmeans_equip_nota.labels_
notas_m2['kmeans_equip_nota'] = notas_m2['kmeans_equip_nota'].astype('category')
notas_m2['kmeans_nota'] = kmeans_nota.labels_
notas_m2['kmeans_nota'] = notas_m2['kmeans_nota'].astype('category')

#%% Análise dos clusters - Classes de Equipamento - KMeans
kmeans_equip_scores = defaultdict(list)
kmeans_equip_scores['Homogeneity'].append(homogeneity_score(notas_m2['classe_equip'], notas_m2['kmeans_equip_nota']))
kmeans_equip_scores['Completeness'].append(completeness_score(notas_m2['classe_equip'], notas_m2['kmeans_equip_nota']))
kmeans_equip_scores['V-measure'].append(v_measure_score(notas_m2['classe_equip'], notas_m2['kmeans_equip_nota']))
kmeans_equip_scores['Silhouette Score'].append(silhouette_score(X_eqnt_bw, notas_m2['kmeans_equip_nota'], metric='euclidean'))
print("KMeans - Classes de Equipamento:")
print(kmeans_equip_scores)

#%% Análise dos clusters - Modos de Falha - KMeans
kmeans_falha_scores = defaultdict(list)
kmeans_falha_scores['Homogeneity'].append(homogeneity_score(notas_m2['modo_falha'], notas_m2['kmeans_nota']))
kmeans_falha_scores['Completeness'].append(completeness_score(notas_m2['modo_falha'], notas_m2['kmeans_nota']))
kmeans_falha_scores['V-measure'].append(v_measure_score(notas_m2['modo_falha'], notas_m2['kmeans_nota']))
kmeans_falha_scores['Silhouette Score'].append(silhouette_score(X_nt_bw, notas_m2['kmeans_nota'], metric='euclidean'))
print("KMeans - Modos de Falha:")
print(kmeans_falha_scores)

#%% Clusterização dos textos - HDBSCAN
from sklearn.cluster import HDBSCAN
hdbscan_equip_nota = HDBSCAN(min_cluster_size=5, metric='euclidean').fit(X_eqnt_bw.toarray())
hdbscan_nota = HDBSCAN(min_cluster_size=5, metric='euclidean').fit(X_nt_bw.toarray())

#%% Adicionando os clusters HDBSCAN às notas classificadas
notas_m2['hdbscan_equip_nota'] = hdbscan_equip_nota.labels_
notas_m2['hdbscan_equip_nota'] = notas_m2['hdbscan_equip_nota'].astype('category')
notas_m2['hdbscan_nota'] = hdbscan_nota.labels_
notas_m2['hdbscan_nota'] = notas_m2['hdbscan_nota'].astype('category')

#%% Análise dos clusters - Classes de Equipamento - HDBSCAN
hdbscan_equip_scores = defaultdict(list)
hdbscan_equip_scores['Homogeneity'].append(homogeneity_score(notas_m2['classe_equip'], notas_m2['hdbscan_equip_nota']))
hdbscan_equip_scores['Completeness'].append(completeness_score(notas_m2['classe_equip'], notas_m2['hdbscan_equip_nota']))
hdbscan_equip_scores['V-measure'].append(v_measure_score(notas_m2['classe_equip'], notas_m2['hdbscan_equip_nota']))
hdbscan_equip_scores['Silhouette Score'].append(silhouette_score(X_eqnt_bw, notas_m2['hdbscan_equip_nota'], metric='euclidean'))
print("HDBSCAN - Classes de Equipamento:")
print(hdbscan_equip_scores)

#%% Análise dos clusters - Modos de Falha - HDBSCAN
hdbscan_falha_scores = defaultdict(list)
hdbscan_falha_scores['Homogeneity'].append(homogeneity_score(notas_m2['modo_falha'], notas_m2['hdbscan_nota']))
hdbscan_falha_scores['Completeness'].append(completeness_score(notas_m2['modo_falha'], notas_m2['hdbscan_nota']))
hdbscan_falha_scores['V-measure'].append(v_measure_score(notas_m2['modo_falha'], notas_m2['hdbscan_nota']))
hdbscan_falha_scores['Silhouette Score'].append(silhouette_score(X_nt_bw, notas_m2['hdbscan_nota'], metric='euclidean'))
print("HDBSCAN - Modos de Falha:")
print(hdbscan_falha_scores)

#%% Classificação das notas com o modelo SVM refinado

from sklearn.pipeline import Pipeline

pipe_c = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=preprocess_text, lowercase=True)),
    ('classifier', SVC(C=1, degree=2, gamma='scale', kernel='linear', random_state=42))
])

pipe_f = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=preprocess_text, lowercase=True)),
    ('classifier', SVC(C=1, degree=2, gamma='scale', kernel='linear', random_state=42))
])

pipe_c.fit(notas_m2['desc_equip'] + ' ' + notas_m2['desc_nota'], notas_m2['classe_equip_novo'])
pipe_f.fit(notas_m2['desc_nota'], notas_m2['modo_falha_novo'])

# Analisando acurácia nos dados de treino
y_c_train_pred = pipe_c.predict(notas_m2['desc_equip'] + ' ' + notas_m2['desc_nota'])
y_f_train_pred = pipe_f.predict(notas_m2['desc_nota'])
train_c_accuracy = accuracy_score(notas_m2['classe_equip_novo'], y_c_train_pred)
train_f_accuracy = accuracy_score(notas_m2['modo_falha_novo'], y_f_train_pred)
print(f"Acurácia no treino - Classes de Equipamento: {train_c_accuracy:.4f}")
print(f"Acurácia no treino - Modos de Falha: {train_f_accuracy:.4f}")

# Classificar as notas com o modelo SVM refinado
notas['classe_equip_pred'] = pipe_c.predict(notas['desc_equip'] + ' ' + notas['desc_nota'])
notas['modo_falha_pred'] = pipe_f.predict(notas['desc_nota'])

#%% Salvando o DataFrame final com as classificações
notas.to_excel(RESULTS_FILE, index=False)