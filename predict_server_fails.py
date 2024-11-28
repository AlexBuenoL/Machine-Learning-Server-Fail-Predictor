"""
    In this script, several classification models are trained in order to predict
    if a server will fail or not in the next month. A set of server data (cpu, io, 
    disk, net, memory and other use parameters) is used for model training.
"""

__author__ = 'Alex Bueno'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from yellowbrick.target.feature_correlation import feature_correlation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import  ConfusionMatrixDisplay,\
                  classification_report,  RocCurveDisplay, PrecisionRecallDisplay,\
                  accuracy_score, f1_score, precision_score, recall_score


# First part: load dataset and data visualization
df = pd.read_csv('tokyo1.tsv', sep="\t")
df.head()

X = df.loc[:, df.columns != 'target']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53, stratify=y)
orig_cols = X_train.columns.to_list() 

X_train.describe().T

# Show the distribution of each variable
fig, axes = plt.subplots(11, 4, figsize=(15, 40))
axes = axes.flatten() 

for i, col in enumerate(X_train.columns):
    ax = axes[i]  
    X_train[col].plot.hist(bins=30, alpha=0.7, ax=ax, title=col)

plt.tight_layout()
plt.show()


X_train_frame = pd.DataFrame(X_train)
X_train_frame.columns=X.columns

# Show correlation matrix between variables
corr = X_train_frame.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, cmap='seismic',  center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5});

plt.figure(figsize=(10,8))
visualizer = feature_correlation(X_train_frame, y_train, labels=list(X_train_frame.columns),method='mutual_info-classification')


# Check if there are null values
print(X_train_frame.isna().sum())


# Dimensionality reduction
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train)

# Variance explained with PCA
fig = plt.figure(figsize=(8,6))
plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_ ,alpha=0.8,marker='.',label="Variancia Explicada")
y_label = plt.ylabel('Variancia explicada')
x_label = plt.xlabel('Componentes')
plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red',marker='.',
         label="Variancia explicada acumulativa")
plt.legend()
plt.title('Porcentaje de variancia explicada por componente')

# Separability visualization with only 2 components
X_trans = pca.transform(X_train)
plt.figure(figsize=(8,8));
sns.scatterplot(x=X_trans[:,0], y=X_trans[:,1], hue=y_train)

# With 3 components
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
cmap = ListedColormap(['red','green'])

scatter = ax.scatter(X_trans[:, 0], X_trans[:, 1], X_trans[:, 2], c=y_train, cmap=cmap, s=50)
colorbar = fig.colorbar(scatter, ax=ax, ticks=[0,1])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()


# t-SNE (2 components)
train_tsne = TSNE(n_components=2, perplexity=10,max_iter=2000, init='random').fit_transform(X_train)
train_tsne = pd.DataFrame(train_tsne, columns=['TSNE1', 'TSNE2'])
train_tsne['class'] = y_train 

fig = plt.figure(figsize=(8,8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='class', data=train_tsne)

# t-SNE (3 components)
train_tsne = TSNE(n_components=3, perplexity=10,max_iter=2000, init='random').fit_transform(X_train)
train_tsne = pd.DataFrame(train_tsne, columns=['TSNE1', 'TSNE2', 'TSNE3'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    train_tsne.TSNE1,
    train_tsne.TSNE2,
    train_tsne.TSNE3,
    c=y_train,  
    cmap='viridis', 
    s=100, 
    depthshade=False
)

colorbar = fig.colorbar(scatter, ax=ax, label='Clases')

ax.set_xlabel('TSNE1')
ax.set_ylabel('TSNE2')
ax.set_zlabel('TSNE3')

plt.show()


# Second part: training classification models

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

gnb.best_score_ = np.mean(cross_val_score(gnb,X_train,y_train,cv=10))
print(f'Acierto de validación cruzada: {gnb.best_score_}')
print(classification_report(y_test, gnb.predict(X_test), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(gnb, X_test,y_test, display_labels=['0', '1'], ax=plt.subplot())

plt.figure(figsize=(8,8))
RocCurveDisplay.from_estimator(gnb, X_test,y_test, pos_label=0, ax=plt.subplot())

# Linear discriminant
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

lda.best_score_ = np.mean(cross_val_score(lda,X_train,y_train,cv=10))
print(f'Acierto de validación cruzada: {lda.best_score_}')
print(classification_report(y_test, lda.predict(X_test), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(lda, X_test,y_test, display_labels=['0', '1'], ax=plt.subplot())

plt.figure(figsize=(8,8))
RocCurveDisplay.from_estimator(lda, X_test,y_test, pos_label=0, ax=plt.subplot())


# Logistic regression
lr = LogisticRegression(max_iter=10000, solver='liblinear')
param = {'penalty':['l1', 'l2'], 'C':10**np.linspace(-3,3,21, endpoint=True)}
lr_gs =  GridSearchCV(lr,param,cv=10, n_jobs=-1, refit=True)
lr_gs.fit(X_train, y_train)

print("Mejores parámetros:", lr_gs.best_params_)
print("Mejor puntuación de validación:", lr_gs.best_score_)
print(classification_report(y_test, lr_gs.best_estimator_.predict(X_test), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(lr_gs.best_estimator_, X_test,y_test, display_labels=['0', '1'], ax=plt.subplot())

plt.figure(figsize=(8,8))
RocCurveDisplay.from_estimator(lr_gs.best_estimator_, X_test,y_test, pos_label=0, ax=plt.subplot())


# Third part: reduce dataset volum woth Recursive Feature Elimination
X_train_df = pd.DataFrame(X_train, columns=orig_cols)
X_test_df = pd.DataFrame(X_test, columns=orig_cols)

rfecv_lda = RFECV(estimator=lda, step=1, cv=10, scoring='accuracy', min_features_to_select=1, n_jobs=-1)
rfecv_lda.fit(X_train_df, y_train)

# Columnas seleccionadas
selected_features_lda = X_train_df.columns[rfecv_lda.support_]
print(f'Número de variables seleccionadas: {len(selected_features_lda)}')
print(f'Lista de las variables seleccionadas: {selected_features_lda.to_list()}')

X_train_lda = X_train_df[selected_features_lda]
X_test_lda = X_test_df[selected_features_lda]

# Re-train the models
reduced_lda = LinearDiscriminantAnalysis()
reduced_lda.fit(X_train_lda, y_train)

reduced_lda.best_score_ = np.mean(cross_val_score(reduced_lda,X_train_lda,y_train,cv=10))
print(f'Acierto de validación cruzada: {reduced_lda.best_score_}')
print(classification_report(y_test, reduced_lda.predict(X_test_lda), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(reduced_lda, X_test_lda, y_test, display_labels=['0', '1'], ax=plt.subplot())

plt.figure(figsize=(8,8))
RocCurveDisplay.from_estimator(reduced_lda, X_test_lda, y_test, pos_label=0, ax=plt.subplot())

# Compare variable weights in original model and the reduced one
selected_indices = [X.columns.get_loc(col) for col in X_train_lda.columns]  
filtered_coefs = lda.coef_[:, selected_indices]  

coefs = pd.DataFrame(filtered_coefs, columns=X_train_lda.columns)
plt.figure(figsize=(40, 2))
sns.heatmap(coefs.abs(), annot=True, linewidths=.5, cbar=True, xticklabels=True, cmap='Blues', annot_kws={'size': 12});
plt.show()

coefs = pd.DataFrame(reduced_lda.coef_)
coefs.columns = X_train_lda.columns
plt.figure(figsize=(40, 2)) 
sns.heatmap(coefs.abs(), annot=True, linewidths=.5, cbar=True, xticklabels=True, cmap='Blues', annot_kws={'size': 12});
plt.show()


rfecv_lr = RFECV(estimator=lr_gs.best_estimator_, step=1, cv=10, scoring='accuracy', min_features_to_select=1, n_jobs=-1)
rfecv_lr.fit(X_train_df, y_train)

# Cols selected
selected_features_lr = X_train_df.columns[rfecv_lr.support_]
print(f'Número de variables seleccionadas: {len(selected_features_lr)}')
print(f'Lista de las variables seleccionadas: {selected_features_lr.to_list()}')

X_train_lr = X_train_df[selected_features_lr]
X_test_lr = X_test_df[selected_features_lr]

reduced_lr = LogisticRegression(max_iter=10000, solver='liblinear')
param = {'penalty':['l1', 'l2'], 'C':10**np.linspace(-3,3,21, endpoint=True)}
reduced_lrgs =  GridSearchCV(reduced_lr,param,cv=10, n_jobs=-1, refit=True)
reduced_lrgs.fit(X_train_lr, y_train)

print("Mejores parámetros:", reduced_lrgs.best_params_)
print("Mejor puntuación de validación:", reduced_lrgs.best_score_)
print(classification_report(y_test, reduced_lrgs.best_estimator_.predict(X_test_lr), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(reduced_lrgs.best_estimator_, X_test_lr, y_test, display_labels=['0', '1'], ax=plt.subplot())

plt.figure(figsize=(8,8))
RocCurveDisplay.from_estimator(reduced_lrgs.best_estimator_, X_test_lr, y_test, pos_label=0, ax=plt.subplot())


selected_indices = [X.columns.get_loc(col) for col in X_train_lr.columns]  
filtered_coefs = lr_gs.best_estimator_.coef_[:, selected_indices]  

# Compare weights between logistic regression models
coefs = pd.DataFrame(filtered_coefs, columns=X_train_lr.columns)
plt.figure(figsize=(12, 2))
sns.heatmap(coefs.abs(), annot=True, linewidths=.5, cbar=True, xticklabels=True, cmap='Blues', annot_kws={'size': 12});
plt.show()

coefs = pd.DataFrame(reduced_lrgs.best_estimator_.coef_)
coefs.columns = X_train_lr.columns
plt.figure(figsize=(12, 2)) 
sns.heatmap(coefs.abs(), annot=True, linewidths=.5, cbar=True, xticklabels=True, cmap='Blues', annot_kws={'size': 12});
plt.show()


# Last part: apply PCA in order to reduce dataset variables and re-evaluate models
pca = PCA(n_components=14)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lda_pca = LinearDiscriminantAnalysis()
lda_pca.fit(X_train_pca, y_train)

lda_pca.best_score_ = np.mean(cross_val_score(lda_pca,X_train_pca,y_train,cv=10))
print(f'Acierto de validación cruzada: {lda_pca.best_score_}')
print(classification_report(y_test, lda_pca.predict(X_test_pca), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(reduced_lrgs.best_estimator_, X_test_lr, y_test, display_labels=cls, ax=plt.subplot())


lr_pca = LogisticRegression(max_iter=10000, solver='liblinear')
param = {'penalty':['l1', 'l2'], 'C':10**np.linspace(-3,3,21, endpoint=True)}
lrgs_pca =  GridSearchCV(lr_pca,param,cv=10, n_jobs=-1, refit=True)
lrgs_pca.fit(X_train_pca, y_train)

print("Mejores parámetros:", lrgs_pca.best_params_)
print("Mejor puntuación de validación:", lrgs_pca.best_score_)
print(classification_report(y_test, lrgs_pca.best_estimator_.predict(X_test_pca), target_names=['0','1']))

plt.figure(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(lrgs_pca.best_estimator_, X_test_pca,y_test, display_labels=['0', '1'], ax=plt.subplot())