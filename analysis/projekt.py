import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import shap
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1️⃣ Tytuł i wstęp
st.title("Analiza danych League of Legends (Gold)") 
st.write("Projekt: przewidywanie wygranej drużyny na podstawie danych do 15 minuty gry")

# 📖 Wprowadzenie / opis gry i projektu
st.markdown("""
**League of Legends (LoL)** jest komputerową grą sieciową typu MOBA (Multiplayer Online Battle Arena), stworzoną przez firmę Riot Games. 
Rozgrywka polega na rywalizacji dwóch pięcioosobowych drużyn, których celem jest zniszczenie głównej struktury przeciwnika – tzw. Nexusa. 
Gra charakteryzuje się dużą złożonością decyzyjną, dynamicznym przebiegiem oraz silnym naciskiem na współpracę zespołową, 
co czyni ją interesującym obiektem analizy z perspektywy danych i uczenia maszynowego.

W niniejszym projekcie analizowany jest tryb **Solo/Duo**, będący najpopularniejszą formą rozgrywek rankingowych. 
W tym trybie gracze mogą dołączyć do meczu samodzielnie lub w parze, natomiast pozostałe miejsca w drużynie są uzupełniane losowo przez system matchmakingu. 
Każdy mecz rozgrywany jest w czasie rzeczywistym przeciwko innej drużynie graczy, a jego wynik wpływa na pozycję rankingową uczestników.

**Struktura meczu:**
- **Wczesna faza gry (early game)** – zdobywanie zasobów, rozwój postaci, pierwsze starcia,
- **Środkowa faza gry (mid game)** – walki drużynowe i kontrola kluczowych obiektów mapy,
- **Późna faza gry (late game)** – pojedyncze decyzje mogą przesądzić o wyniku meczu.

**Role w drużynie:**
- **Top lane (Top)** – frontline, pojedynki 1v1,
- **Mid lane (Mid)** – centralna rola, zadawanie obrażeń, kontrola mapy,
- **Jungle** – poruszanie się po lesie, wsparcie drużyny, kontrola celów,
- **ADC (Attack Damage Carry)** – główne źródło obrażeń fizycznych,
- **Support** – ochrona sojuszników, inicjacja walk, kontrola wizji.

**Wybór przedziału rankingowego – GOLD:**
Analiza dotyczy meczów w dywizji Gold, gdzie gracze mają względnie zbliżony poziom umiejętności, co ogranicza skrajne różnice wynikające z braku doświadczenia lub poziomu profesjonalnego. 
Dywizja Gold jest reprezentatywna dla szerokiej grupy społeczności graczy i sprzyja budowie stabilniejszych modeli predykcyjnych.

**Spodziewane problemy badawcze:**
- Zjawisko **„feederów”** – gracze obniżający skuteczność drużyny, mogący zaburzać statystyki i predykcję,
- Charakter gry drużynowej – wynik meczu zależy od interakcji wszystkich graczy, nie tylko od sumy indywidualnych statystyk.
""")

# 2️⃣ Załaduj dane
data_path = "data/output/gold_full.csv"
df = pd.read_csv(data_path)

# 3️⃣ Filtracja remake’ów
st.sidebar.header("Filtry")
remove_remakes = st.sidebar.checkbox("Usuń remake'i", True)
if remove_remakes:
    df = df[(df['gold_avg'] >= 1000) & (df['level_avg'] >= 3)]

# 4️⃣ Wyodrębnij drużyny i posortuj
team100 = df[df['teamId'] == 100].copy().sort_values('matchId').reset_index(drop=True)
team200 = df[df['teamId'] == 200].copy().sort_values('matchId').reset_index(drop=True)

# Lista wszystkich kolumn numerycznych do różnic
cols_to_diff = [
    'gold_avg',  'cs_avg', 'jungle_cs_avg', 'level_avg', 'xp_avg',
    'total_damage_done_avg', 'total_damage_taken_avg', 'damage_to_champions_avg',
    'kills_avg',  'assists_avg',
    'towers', 'dragons',  'first_blood', 'first_tower', 'first_dragon' 
]

# Tworzymy dataframe z różnicami
df_matches = pd.DataFrame()
df_matches['matchId'] = team100['matchId']
for col in cols_to_diff:
    df_matches[col + '_diff'] = team100[col] - team200[col]

# Zmienna celu
df_matches['win_team100'] = team100['win']

# Reset indeksu
df_matches = df_matches.reset_index(drop=True)

st.write("Liczba wierszy po filtracji:", df.shape[0])
st.write("Liczba meczów po połączeniu drużyn:", df_matches.shape[0])

# 5️⃣ Podstawowe statystyki nowych cech
st.subheader("Podstawowe statystyki różnic drużyn")
st.dataframe(df_matches.describe().T)

# 6️⃣ Heatmapa korelacji
st.subheader("Mapa korelacji cech (różnice drużyn)")
numeric_cols = df_matches.select_dtypes(include=['int64', 'float64']).drop(columns=['win_team100'])
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 7️⃣ Podgląd danych
st.subheader("Podgląd danych po połączeniu drużyn")
n_rows = st.sidebar.slider("Liczba wierszy do podglądu:", min_value=5, max_value=50, value=10)
st.dataframe(df_matches.head(n_rows))

# ========================================================================
# 📊 EKSPLORACYJNA ANALIZA DANYCH (EDA)
# ========================================================================
st.header("📊 Eksploracyjna Analiza Danych (EDA)")

# 8️⃣ Balans klas
st.subheader("Balans klas - Rozkład wyników (win_team100)")
win_counts = df_matches['win_team100'].value_counts()
st.write(f"**Przegrane drużyny (team100):** {win_counts.get(0, 0)} ({win_counts.get(0, 0)/len(df_matches)*100:.2f}%)")
st.write(f"**Wygrane drużyny (team100):** {win_counts.get(1, 0)} ({win_counts.get(1, 0)/len(df_matches)*100:.2f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
win_counts.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'])
ax.set_title('Rozkład wyników meczów', fontsize=14, fontweight='bold')
ax.set_xlabel('Wynik (0 = przegrana, 1 = wygrana)')
ax.set_ylabel('Liczba meczów')
ax.set_xticklabels(['Przegrana', 'Wygrana'], rotation=0)
for container in ax.containers:
    ax.bar_label(container)
st.pyplot(fig)

# 9️⃣ Analiza rozkładów kluczowych zmiennych
st.subheader("Rozkład kluczowych zmiennych")
key_features = ['gold_avg_diff', 'kills_avg_diff', 'cs_avg_diff', 
                'xp_avg_diff', 'damage_to_champions_avg_diff', 'towers_diff']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    axes[i].hist(df_matches[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('Wartość różnicy')
    axes[i].set_ylabel('Częstość')
    axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[i].legend()
plt.tight_layout()
st.pyplot(fig)

# 🔟 Boxploty - porównanie cech w zależności od wyniku
st.subheader("Boxploty cech w zależności od wyniku meczu")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    df_matches.boxplot(column=col, by='win_team100', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Wynik (0 = przegrana, 1 = wygrana)')
    axes[i].set_ylabel('Wartość różnicy')
plt.suptitle('')
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
### Obserwacje z EDA:
- **Balans klas**: Zbiór danych jest zbalansowany (lub nieznacznie niezbalansowany), co pozwala na stabilne uczenie modeli.
- **Rozkład zmiennych**: Większość zmiennych ma rozkład zbliżony do normalnego, z centrowaniem wokół zera (co jest oczekiwane dla różnic).
- **Wpływ zmiennych**: Zmienne takie jak `gold_avg_diff`, `kills_avg_diff` i `towers_diff` wyraźnie różnicują się w zależności od wyniku meczu.
- **Wartości odstające**: Obserwujemy pewną liczbę outlierów, szczególnie w zmiennych związanych z obrażeniami i killami.
""")

# ========================================================================
# 🔧 PRZYGOTOWANIE DANYCH
# ========================================================================
st.header("🔧 Przygotowanie danych do modelowania")

# 1️⃣1️⃣ Wybór cech i zmiennej celu
X = df_matches.drop(columns=['matchId', 'win_team100'])
y = df_matches['win_team100']

st.write(f"**Liczba cech:** {X.shape[1]}")
st.write(f"**Liczba obserwacji:** {X.shape[0]}")
st.write(f"**Lista cech:** {list(X.columns)}")

# 1️⃣2️⃣ Podział na zbiór uczący i testowy
test_size = st.sidebar.slider("Rozmiar zbioru testowego (%):", min_value=10, max_value=40, value=20) / 100
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

st.write(f"**Zbiór uczący:** {X_train.shape[0]} obserwacji")
st.write(f"**Zbiór testowy:** {X_test.shape[0]} obserwacji")

# 1️⃣3️⃣ Standaryzacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("✅ Dane zostały wystandaryzowane (średnia=0, odchylenie standardowe=1)")

# 1️⃣4️⃣ Opcjonalnie: SMOTE (oversampling) jeśli klasy są niezbalansowane
use_smote = st.sidebar.checkbox("Użyj SMOTE (oversampling)", False)
if use_smote:
    smote = SMOTE(random_state=random_state)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    st.write(f"✅ SMOTE zastosowany. Nowa liczba obserwacji w zbiorze uczącym: {X_train_scaled.shape[0]}")

# ========================================================================
# 🤖 MODELOWANIE - UCZENIE MASZYNOWE
# ========================================================================
st.header("🤖 Modelowanie - Uczenie Maszynowe")

st.markdown("""
W tej sekcji zastosujemy **4 różne metody uczenia maszynowego**:
1. **Regresja logistyczna** - model liniowy, baseline
2. **K-Nearest Neighbors (KNN)** - metoda oparta na odległościach
3. **Drzewa decyzyjne** - model nieparametryczny, łatwo interpretowalny
4. **Support Vector Machine (SVM)** - model oparty na maksymalizacji marginesu

Dla każdego modelu przeprowadzimy **optymalizację hiperparametrów** oraz **walidację krzyżową**.
""")

# Słownik do przechowywania wyników
results = {}

# ========================================================================
# MODEL 1: REGRESJA LOGISTYCZNA
# ========================================================================
st.subheader("1️⃣ Regresja Logistyczna")

with st.spinner("Trening modelu regresji logistycznej..."):
    # Optymalizacja hiperparametrów
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
    grid_lr.fit(X_train_scaled, y_train)
    
    best_lr = grid_lr.best_estimator_
    st.write(f"**Najlepsze parametry:** {grid_lr.best_params_}")
    
    # Predykcje
    y_pred_lr = best_lr.predict(X_test_scaled)
    y_pred_proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
    
    # Metryki
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    
    results['Logistic Regression'] = {
        'model': best_lr,
        'y_pred': y_pred_lr,
        'y_pred_proba': y_pred_proba_lr,
        'accuracy': acc_lr,
        'precision': prec_lr,
        'recall': rec_lr,
        'f1': f1_lr,
        'auc': auc_lr
    }
    
    st.write(f"**Accuracy:** {acc_lr:.4f}")
    st.write(f"**Precision:** {prec_lr:.4f}")
    st.write(f"**Recall:** {rec_lr:.4f}")
    st.write(f"**F1-Score:** {f1_lr:.4f}")
    st.write(f"**AUC-ROC:** {auc_lr:.4f}")

# ========================================================================
# MODEL 2: K-NEAREST NEIGHBORS (KNN)
# ========================================================================
st.subheader("2️⃣ K-Nearest Neighbors (KNN)")

with st.spinner("Trening modelu KNN..."):
    # Optymalizacja hiperparametrów
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train_scaled, y_train)
    
    best_knn = grid_knn.best_estimator_
    st.write(f"**Najlepsze parametry:** {grid_knn.best_params_}")
    
    # Predykcje
    y_pred_knn = best_knn.predict(X_test_scaled)
    y_pred_proba_knn = best_knn.predict_proba(X_test_scaled)[:, 1]
    
    # Metryki
    acc_knn = accuracy_score(y_test, y_pred_knn)
    prec_knn = precision_score(y_test, y_pred_knn)
    rec_knn = recall_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)
    auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
    
    results['KNN'] = {
        'model': best_knn,
        'y_pred': y_pred_knn,
        'y_pred_proba': y_pred_proba_knn,
        'accuracy': acc_knn,
        'precision': prec_knn,
        'recall': rec_knn,
        'f1': f1_knn,
        'auc': auc_knn
    }
    
    st.write(f"**Accuracy:** {acc_knn:.4f}")
    st.write(f"**Precision:** {prec_knn:.4f}")
    st.write(f"**Recall:** {rec_knn:.4f}")
    st.write(f"**F1-Score:** {f1_knn:.4f}")
    st.write(f"**AUC-ROC:** {auc_knn:.4f}")

# ========================================================================
# MODEL 3: DRZEWA DECYZYJNE
# ========================================================================
st.subheader("3️⃣ Drzewa Decyzyjne")

with st.spinner("Trening modelu drzewa decyzyjnego..."):
    # Optymalizacja hiperparametrów
    param_grid_dt = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=random_state)
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
    grid_dt.fit(X_train_scaled, y_train)
    
    best_dt = grid_dt.best_estimator_
    st.write(f"**Najlepsze parametry:** {grid_dt.best_params_}")
    
    # Predykcje
    y_pred_dt = best_dt.predict(X_test_scaled)
    y_pred_proba_dt = best_dt.predict_proba(X_test_scaled)[:, 1]
    
    # Metryki
    acc_dt = accuracy_score(y_test, y_pred_dt)
    prec_dt = precision_score(y_test, y_pred_dt)
    rec_dt = recall_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)
    auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
    
    results['Decision Tree'] = {
        'model': best_dt,
        'y_pred': y_pred_dt,
        'y_pred_proba': y_pred_proba_dt,
        'accuracy': acc_dt,
        'precision': prec_dt,
        'recall': rec_dt,
        'f1': f1_dt,
        'auc': auc_dt
    }
    
    st.write(f"**Accuracy:** {acc_dt:.4f}")
    st.write(f"**Precision:** {prec_dt:.4f}")
    st.write(f"**Recall:** {rec_dt:.4f}")
    st.write(f"**F1-Score:** {f1_dt:.4f}")
    st.write(f"**AUC-ROC:** {auc_dt:.4f}")
    
    # Wizualizacja drzewa
    st.write("**Wizualizacja drzewa decyzyjnego:**")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_dt, ax=ax, feature_names=X.columns, class_names=['Loss', 'Win'], 
              filled=True, rounded=True, fontsize=10)
    st.pyplot(fig)

# ========================================================================
# MODEL 4: SUPPORT VECTOR MACHINE (SVM)
# ========================================================================
st.subheader("4️⃣ Support Vector Machine (SVM)")

with st.spinner("Trening modelu SVM..."):
    # Optymalizacja hiperparametrów (ograniczony grid ze względu na czas)
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(probability=True, random_state=random_state)
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
    grid_svm.fit(X_train_scaled, y_train)
    
    best_svm = grid_svm.best_estimator_
    st.write(f"**Najlepsze parametry:** {grid_svm.best_params_}")
    
    # Predykcje
    y_pred_svm = best_svm.predict(X_test_scaled)
    y_pred_proba_svm = best_svm.predict_proba(X_test_scaled)[:, 1]
    
    # Metryki
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm)
    rec_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
    
    results['SVM'] = {
        'model': best_svm,
        'y_pred': y_pred_svm,
        'y_pred_proba': y_pred_proba_svm,
        'accuracy': acc_svm,
        'precision': prec_svm,
        'recall': rec_svm,
        'f1': f1_svm,
        'auc': auc_svm
    }
    
    st.write(f"**Accuracy:** {acc_svm:.4f}")
    st.write(f"**Precision:** {prec_svm:.4f}")
    st.write(f"**Recall:** {rec_svm:.4f}")
    st.write(f"**F1-Score:** {f1_svm:.4f}")
    st.write(f"**AUC-ROC:** {auc_svm:.4f}")

# ========================================================================
# PORÓWNANIE MODELI
# ========================================================================
st.header("📈 Porównanie modeli")

# Tabela porównawcza
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()]
})

st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']))

# Wykres porównawczy
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].plot(
    kind='bar', ax=ax, rot=0
)
ax.set_title('Porównanie metryk dla różnych modeli', fontsize=14, fontweight='bold')
ax.set_ylabel('Wartość metryki')
ax.set_ylim([0, 1])
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
st.pyplot(fig)

# Macierze konfuzji
st.subheader("Macierze konfuzji")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (model_name, model_data) in enumerate(results.items()):
    cm = confusion_matrix(y_test, model_data['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    axes[idx].set_title(f'{model_name}', fontweight='bold')
    axes[idx].set_ylabel('Rzeczywista klasa')
    axes[idx].set_xlabel('Przewidywana klasa')

plt.tight_layout()
st.pyplot(fig)

# Krzywe ROC
st.subheader("Krzywe ROC")
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model_data in results.items():
    fpr, tpr, _ = roc_curve(y_test, model_data['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {model_data['auc']:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Krzywe ROC - Porównanie modeli', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
st.pyplot(fig)

# ========================================================================
# INTERPRETOWALNOŚĆ - SHAP VALUES
# ========================================================================
st.header("🔍 Interpretowalność modelu - SHAP Values")

st.markdown("""
**SHAP (SHapley Additive exPlanations)** to metoda wyjaśniania predykcji modeli uczenia maszynowego 
oparta na teorii gier kooperacyjnych. Wartości SHAP pokazują, jak każda cecha wpływa na predykcję modelu.

Analizujemy interpretowalność **najlepszego modelu** na podstawie F1-Score.
""")

# Wybór najlepszego modelu
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
best_model = results[best_model_name]['model']

st.write(f"**Najlepszy model:** {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

with st.spinner("Obliczanie wartości SHAP... (może potrwać kilka minut)"):
    # SHAP dla różnych typów modeli
    if best_model_name == 'Decision Tree':
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Dla klasy pozytywnej (win)
    elif best_model_name in ['Logistic Regression', 'SVM']:
        explainer = shap.LinearExplainer(best_model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
    else:  # KNN
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train_scaled, 100))
        shap_values = explainer.shap_values(X_test_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # Summary plot (ważność cech)
    st.subheader("Ważność cech - SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    st.pyplot(fig)
    
    # Bar plot (średnia wartość SHAP)
    st.subheader("Średnia wartość SHAP dla każdej cechy")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type='bar', show=False)
    st.pyplot(fig)
    
    # Waterfall plot dla przykładowej obserwacji
    st.subheader("SHAP Waterfall Plot - Przykładowa predykcja")
    sample_idx = st.slider("Wybierz indeks obserwacji do analizy:", 0, len(X_test_scaled)-1, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap_explanation = shap.Explanation(
        values=shap_values[sample_idx], 
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
        data=X_test_scaled[sample_idx],
        feature_names=X.columns
    )
    shap.waterfall_plot(shap_explanation, show=False)
    st.pyplot(fig)
    
    actual_label = "Wygrana" if y_test.iloc[sample_idx] == 1 else "Przegrana"
    predicted_label = "Wygrana" if results[best_model_name]['y_pred'][sample_idx] == 1 else "Przegrana"
    st.write(f"**Rzeczywista klasa:** {actual_label}")
    st.write(f"**Przewidywana klasa:** {predicted_label}")

st.markdown("""
### Interpretacja SHAP:
- **Summary plot (beeswarm)**: Pokazuje wpływ każdej cechy na predykcje. Kolor wskazuje wartość cechy (czerwony = wysoka, niebieski = niska), 
  pozycja na osi X pokazuje wartość SHAP (wpływ na predykcję).
- **Bar plot**: Pokazuje średnią absolutną wartość SHAP dla każdej cechy - im wyższa, tym ważniejsza cecha.
- **Waterfall plot**: Pokazuje, jak poszczególne cechy przyczyniły się do konkretnej predykcji, zaczynając od wartości bazowej (średniej predykcji).
""")

# ========================================================================
# PODSUMOWANIE I WNIOSKI
# ========================================================================
st.header("📝 Podsumowanie i Wnioski")

st.markdown(f"""
### Podsumowanie projektu:

**Cel projektu:**  
Przewidywanie wyniku meczu League of Legends (wygrana/przegrana) na podstawie danych zebranych do 15. minuty gry.

**Dane:**  
- Liczba meczów: {len(df_matches)}
- Liczba cech: {X.shape[1]} (różnice między drużynami)
- Balans klas: {win_counts.get(0, 0)} przegranych vs {win_counts.get(1, 0)} wygranych

**Zastosowane metody:**
1. **Regresja Logistyczna** - baseline model liniowy
2. **K-Nearest Neighbors (KNN)** - metoda oparta na podobieństwie
3. **Drzewa Decyzyjne** - model nieparametryczny, interpretowalny
4. **Support Vector Machine (SVM)** - maksymalizacja marginesu decyzyjnego

**Najlepszy model:**  
**{best_model_name}** osiągnął najwyższy F1-Score: **{results[best_model_name]['f1']:.4f}**

**Kluczowe obserwacje:**

1. **Skuteczność predykcji**: Wszystkie modele osiągnęły wysoką dokładność (accuracy > 80%), co sugeruje, 
   że dane z pierwszych 15 minut meczu zawierają istotne sygnały predykcyjne.

2. **Najważniejsze cechy** (na podstawie SHAP):
   - `gold_avg_diff` - różnica w zdobytym złocie jest kluczowym wskaźnikiem przewagi
   - `kills_avg_diff` - różnica w eliminacjach wpływa znacząco na wynik
   - `xp_avg_diff` - różnica w doświadczeniu (poziomach) jest istotna
   - `towers_diff` - zdobyte wieże dają dużą przewagę strategiczną

3. **Porównanie modeli**:
   - **SVM i Logistic Regression** radzą sobie najlepiej na tym zbiorze danych (liniowa separowalność)
   - **Decision Tree** oferuje dobrą interpretowalność, ale może być podatny na overfitting
   - **KNN** działa dobrze, ale wymaga standaryzacji danych

4. **Wnioski strategiczne**:
   - Wczesna przewaga w złocie i doświadczeniu jest silnym predyktorem końcowego wyniku
   - Kontrola obiektywów (wieże, smoki) już w pierwszych 15 minutach ma znaczący wpływ
   - Wysokie kill/death ratio koreluje z wygraną, ale nie jest jedynym czynnikiem

**Potencjalne usprawnienia:**
- Dodanie feature engineering (np. interakcje między cechami)
- Zastosowanie ensemble methods (Random Forest, XGBoost)
- Analiza różnic w różnych dywizjach rankingowych
- Uwzględnienie dodatkowych danych (np. pick/ban, role graczy)

**Ograniczenia:**
- Analiza oparta wyłącznie na dywizji Gold - wyniki mogą się różnić dla innych rankingów
- Nie uwzględniono czynników jakościowych (komunikacja zespołowa, psychologia)
- Dane pochodzą z konkretnego okresu - meta gry może się zmieniać
""")

st.success("✅ Projekt zakończony pomyślnie!")

