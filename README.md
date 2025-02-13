# Machine-Learning-project-hw

# 📊 Beadandó – Hallgatók Fizetési Szokásai

## 📌 Leírás
Ez a projekt a hallgatók költési szokásait vizsgálja meg különböző demográfiai csoportok és tanulmányi háttér szerint. A vizsgált adatkészlet **1000 diák** fiktív költési adatait tartalmazza, beleértve a havi jövedelmüket, pénzügyi támogatásukat, különböző kiadási kategóriákat és preferált fizetési módjukat.

## 🗂 Adatkészlet
A **student_spending.csv** fájl tartalmazza a következő oszlopokat:
- **Age** – Hallgató életkora (ev)
- **Gender** – Nem (Male, Female, Non-binary)
- **Year in School** – Tanulmányi év (Freshman, Sophomore, Junior, Senior)
- **Major** – Szak
- **Monthly Income** – Havi jövedelem ($)
- **Financial Aid** – Pénzügyi támogatás ($)
- **Kiadások**:
  - Tuition (tandíj)
  - Housing (lakhatás)
  - Food (étel)
  - Transportation (közlekedés)
  - Books & Supplies (könyvek, kellékek)
  - Entertainment (szórakozás)
  - Personal Care (személyes gondozás)
  - Technology (technológia)
  - Health & Wellness (egészség)
  - Miscellaneous (egyéb kiadások)
- **Preferred Payment Method** – Preferált fizetési mód (Cash, Credit/Debit Card, Mobile Payment App)

## 🎯 Cél
A projekt célja a hallgatók **preferált fizetési mód szerinti csoportosítása** (készpénz vs. mobil/kártyás fizetés). Ehhez **gépi tanulási osztályozó algoritmusokat** alkalmazok, amelyek segítségével megvizsgálható, milyen demográfiai jellemzők és pénzügyi mutatók befolyásolják a fizetési preferenciákat.

## 🛠 Technológiák
- **Python**: adatelemzés és modelltréning
- **pandas**: adatok kezelése
- **scikit-learn**: gépi tanulási modellek
- **matplotlib, seaborn**: vizualizáció

## 🏗 Modelltréning
A hallgatók fizetési módjának osztályozásához **Döntési fa**, **Random Forest** és **XGBoost** modelleket használtam.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Adatok beolvasása
data = pd.read_csv('student_spending.csv')
X = data.drop('Preferred Payment Method', axis=1)
y = data['Preferred Payment Method']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell tanítása
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Előrejelzés és pontosság ellenőrzés
y_pred = clf.predict(X_test)
print("Pontosság:", accuracy_score(y_test, y_pred))
```

## 📊 Eredmények
A legjobb teljesítényt az **XGBoost** modell érte el **35,87%-os pontossággal**.
- A **magasabb jövedelmű hallgatók** gyakrabban használnak mobil- vagy kártyás fizetést.
- A **felsőbb éves hallgatók** kevésbé használnak készpénzt.
- A **művészeti és humán tudományok hallgatói** gyakrabban fizetnek készpénzzel, mint a STEM hallgatók.



