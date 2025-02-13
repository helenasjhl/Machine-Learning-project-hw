# -*- coding: utf-8 -*-
"""VPMNQN Beadandó ml.

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i06qNQaZM8Hp96kh7Sy9-wYXefXaWCWj

Könyvtárak importálása:
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import numpy as np

"""Beolvassuk fájlt"""

student_spending = pd.read_csv('xstudent_spending.csv')
student_spending

#átalakítjuk az oszlopunk értékeit hogy tudjunk vele dologzni
student_spending["preferred_payment_method"] = student_spending["preferred_payment_method"].str.replace('Cash', '0')
student_spending["preferred_payment_method"] = student_spending["preferred_payment_method"].str.replace('Credit/Debit Card', '1')
student_spending["preferred_payment_method"] = student_spending["preferred_payment_method"].str.replace('Mobile Payment App', '2')
student_spending["preferred_payment_method"] = student_spending["preferred_payment_method"].astype(float)
student_spending.info()

# Kategorikus oszlopok átalakítása numerikus értékekké
label_encoder = LabelEncoder()
student_spending['gender'] = label_encoder.fit_transform(student_spending['gender'])
student_spending['year_in_school'] = label_encoder.fit_transform(student_spending['year_in_school'])
student_spending['major'] = label_encoder.fit_transform(student_spending['major'])

# @title preferred_payment_method

from matplotlib import pyplot as plt
import seaborn as sns
student_spending.groupby('preferred_payment_method').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

"""Az alábbi diagrammon, lláthatjuk, hogy hasonló számban oszlanak el a fizetési szokások."""

#Kidobjuk azokat a sorokat, ahol duplikált az érték.
student_spending=student_spending.drop_duplicates()
print('Shape After deleting duplicate values:', student_spending.shape)

#Kidobjuk azokat a sorokat, ahol nincs érték.
student_spending.dropna(inplace=True)
print('Shape After deleting missing values:', student_spending.shape)

"""### **A modell tanítása**"""

# ez a függvény a modell teljesítményét adja vissza (accuracy, classification report és confusion matrix)
def model_performance(y_test,y_pred, method):

    # calculate the accruacy of the model
    print("Accuracy score of the model", accuracy_score(y_test,y_pred))
    print("Classification report \n")

    #generate the classification report
    print(classification_report(y_test,y_pred))

    #generate the confusion matrix

    cnf_matrix_log = confusion_matrix(y_test, y_pred)

    print(cnf_matrix_log)

# Felosztás tanító és teszt adatokra
X = student_spending.drop(['preferred_payment_method'], axis=1)  # Független változók
y = student_spending['preferred_payment_method']  # Célváltozó
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""1. Döntési fa osztályozó modell"""

# Döntési fa osztályozó modell létrehozása és tanítása
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

"""Amikor az algoritmus véletlenszerű döntéseket hoz, például az adatok megosztása a tanító- és tesztadatok között, vagy a fa csomópontjainak kiválasztása, a random_state beállítása garantálja, hogy ugyanazokat a véletlenszerű eredményeket kapjuk minden egyes futásnál. Ez kiszámíthatóvá és reprodukálhatóvá teszi az eredményeket."""

# Tesztelés a modell pontosságának ellenőrzésére
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Pontosság:", accuracy)

# Osztályozási jelentés és konfúziós mátrix
print("\nOsztályozási jelentés:")
print(classification_report(y_test, y_pred))

print("\nKonfúziós mátrix:")
print(confusion_matrix(y_test, y_pred))

"""2. **Random forest** osztályozó modell


"""

# paraméterezés
model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50,max_leaf_nodes = 30)
# fitting the model
model_rf.fit(X_train, y_train)

# make predictions
prediction_test = model_rf.predict(X_test)

# a modell teljesítményének értékelés
model_performance(y_test,prediction_test, 'Random Forest')

"""A modellek értékelésére és finomhangolására keresztezéses validációt használhatunk, hogy objektíven értékelhessük a modellek teljesítményét"""

# keresztvalidáció
from sklearn.model_selection import RepeatedStratifiedKFold
# apply k-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_rf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

"""Modell hangolása A RandomForest osztályozó teljesítményének optimalizálásához a különböző hiperparaméterek értékein un. rácsos keresést alkalmazhatunk. A hiperparaméter-értékek különböző kombinációinak szisztematikus vizsgálatával megtalálhatjuk azt a konfigurációt, amely javítja a modell előrejelző képességeit. Ez a folyamat lehetővé teszi számunkra a RandomForest osztályozó finomhangolását és a paraméterbeállítások optimális egyensúlyának elérését, ami végső soron javítja az általános pontosságát és robusztusságát.
A GridSearchCV segítségével megadhatjuk ezeket a lehetséges értékeket, és a GridSearchCV végigpróbál minden lehetséges kombinációt, majd megtalálja a legjobbat.
"""

# a paraméter rács randomizált kialakítása
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'n_estimators': [200, 500],
    'max_leaf_nodes' : [20, 30]
}

# A grid search modell indítása
grid_search = GridSearchCV(estimator = model_rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)

grid_search.best_params_

"""Feature Importance Scores (változók fontossága)
A Random Forest a jellemző fontosságát annak értékelésével számítják ki, hogy az egyes jellemzők mennyire járulnak hozzá a keveredés (osztályozásban) vagy a hiba (regresszióban) csökkentéséhez, amikor a döntési fák együttesén belüli felosztási kritériumként használják őket. Azokat a jellemzőket, amelyek következetesen tisztább csomópontokhoz vagy alacsonyabb hibához vezetnek, fontosabbnak tekintik, és a jellemzők kiválasztásának alapjául szolgálnak.
"""

forest_importances = pd.Series(model_rf.feature_importances_, index=X.columns.values)
std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

"""A legfontosabb változók:
1.   food
2.   entertainment
3.   technology
4.   miscellaneous

3. XGBOOST

Extreme Gradient Boosting, erőteljes gépi tanuló algortimus, amely klasszifikációs és regressziós feladatokra is használható. Az algoritmus alpja a döntési fa módszer, az XGBoost több un. gyenge tanuló algortimust épít fel, és ezekkel folyamatosan javítja modell pontosságát. Megvizsgáljuk, hogy az XGBOOST jobb eredményt ad-e, mint a Random Forest.
"""

# A változók sztenderdizálása
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# egydimenziós vektorrá alakítás
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

# Az XGBClassifier importja
from xgboost import XGBClassifier

# A modell tanítása
model_xgb = XGBClassifier(learning_rate=0.01,
                            n_estimators=50,
                            max_depth=6,
                            random_state=42,
                            n_jobs=-1)
model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_test)

model_performance(y_test,y_pred, 'XGBoost')



""" Nincs javulás a Random  foresthez képest."""

# gridsearch
# grid search for the XGBClassifier
clf = GridSearchCV(
        model_xgb,
        {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200]},
        verbose=1,
        n_jobs=2,
    )
clf.fit(X_train,y_train)
print(clf.best_score_)
print(clf.best_params_)

"""Mivel a célváltozó több mint két osztályból áll (többosztályos probléma), és az ROC AUC (Receiver Operating Characteristic Area Under the Curve) érték csak bináris osztályozásra használható. Az ROC AUC számítása olyan problémákra van tervezve, ahol csak két osztály van.

Ha a célváltozóm többosztályos, akkor más értékelési metrikát kell használni a keresztvalidáció során. Például az egyik lehetőség a accuracy, ami az osztályozási pontosságot méri.
"""

# Az "preferred_payment_method" oszlop értékeinek átalakítása binárisra
student_spending["preferred_payment_method"] = (student_spending["preferred_payment_method"] == "Cash").astype(int)

# Átalakított adatok ellenőrzése
#print(student_spending.head())

from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier

# XGBoost osztály definiálása
model_xgb = XGBClassifier(objective="binary:logistic", colsample_bytree=0.3, learning_rate=0.1,
                           max_depth=5, alpha=10)

# Keresztvalidáció
scores = cross_val_score(model_xgb, X, y, cv=3, scoring='accuracy')

# Eredmények kiírása
print("Pontosság átlagos értéke:", scores.mean())

"""
# **Következtetés**:

A vizsgált adatokon végzett modellezés során több algoritmust is kipróbáltam, köztük a Döntési Fa, Random Forest és XGBoost modelleket. Az eredmények alapján mindhárom modell hasonló pontosságot mutatott, körülbelül 33-37% körüli értéket. Az adatok előfeldolgozása és a hiperparaméterek hangolása sem vezetett jelentős javuláshoz a modell teljesítményében.

A feature importance elemzés során megfigyelhető, hogy az étel, szórakozás, technológia és egyéb kiadások jelentős befolyással bírnak a preferált fizetési módszerre. Ezek az eredmények érthetőek és életképesek lehetnek egy diák szempontjából, és arra utalnak, hogy a modellünk megfelelően figyelembe veszi ezeket a tényezőket.

Az eddigi eredmények alapján a jelenlegi modellek még nem rendelkeznek elég pontos előrejelzéssel a preferált fizetési módszer meghatározására."""