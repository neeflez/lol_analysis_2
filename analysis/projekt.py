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


# Tytuł i wstęp
st.title("Analiza danych League of Legends z dywizji Gold") 
st.write("Projekt: przewidywanie wygranej drużyny na podstawie danych do 15 minuty gry")

# Wprowadzenie / opis gry i projektu
st.markdown("""
**League of Legends (LoL)** jest komputerową grą sieciową typu MOBA (Multiplayer Online Battle Arena), stworzoną przez firmę Riot Games. 
Rozgrywka polega na rywalizacji dwóch pięcioosobowych drużyn, których celem jest zniszczenie głównej struktury przeciwnika – tzw. Nexusa. 
Gra charakteryzuje się dużą złożonością decyzyjną, dynamicznym przebiegiem oraz silnym naciskiem na współpracę zespołową, 
co czyni ją interesującym obiektem analizy z perspektywy danych i uczenia maszynowego.

Każdy gracz przed rozpoczęciem meczu wybiera jedną z 170 postaci. Każda postać charakteryzuje się innymi umiejętnościami, stylem gry oraz rodzajem zadawanych obrażeń. 
Niektóre postacie współgrają ze sobą lepiej, inne gorzej, a niektóre mają przewagę nad innymi postaciami – tzw. counter postacie.

Kombinacje wyboru postaci mają większe znaczenie na wyższych poziomach rozgrywek. W tym projekcie analizowane są mecze w przedziale rang **Gold**, który nie jest wysoki, więc wpływ wyboru postaci można uznać za mniej istotny.  
Dywizje w League of Legends w kolejności od najniższej do najwyższej to:  
Iron, Bronze, Silver, Gold, Platinum, Emerald, Diamond, Master, Challenger.

Umiejętności graczy dzielimy na dwie kategorie: **mikro** i **makro**.  
- **Mikro** – umiejętności kontroli postaci oraz wiedza o postaciach przeciwników.  
- **Makro** – wszystko inne, czyli rozumienie gry jako całości: poruszanie się po mapie, zdobywanie celów, zarządzanie falami minionów i podejmowanie decyzji w drużynie.

Na niskim poziomie rozgrywek większe znaczenie mają umiejętności makro, dlatego w projekcie skupiamy się na statystykach drużynowych, takich jak złoto, zabójstwa i kontrola mapy.

**Szumy w danych:**  
- Pojedynczy gracze zbyt dobrzy lub zbyt słabi mogą zaburzać działanie modelu, wpływając na przewidywania wyniku meczu.  
- Na niższych rangach zdarzają się nietypowe sytuacje, np. błędne decyzje kilku graczy w jednym momencie, które mogą całkowicie zmienić wynik meczu.  

W niniejszym projekcie analizowany jest tryb **Solo/Duo**, w którym gracze mogą dołączyć do meczu samodzielnie lub w parze, a pozostałe miejsca w drużynie uzupełnia system matchmakingu.  
Każdy mecz rozgrywany jest przeciwko innej drużynie, a jego wynik wpływa na pozycję rankingową uczestników.

**Struktura meczu:**
- **Early game** – zdobywanie zasobów, rozwój postaci, pierwsze starcia.  
- **Mid game** – walki drużynowe i kontrola kluczowych obiektów mapy.  
- **Late game** – pojedyncze decyzje mogą przesądzić o wyniku meczu.

Standardowy mecz League of Legends na mapie Summoner’s Rift trwa średnio **25–30 minut**. W niniejszym projekcie analizowane są **dane do 15. minuty gry**, ponieważ:  
- w tym czasie najczęściej kształtuje się przewaga jednej z drużyn,  
- w 15. minucie rozgrywki gracze uzyskują możliwość głosowania nad poddaniem meczu (surrender).  

Analiza danych do 15. minuty pozwala zatem ocenić przewagę drużyny w kluczowym momencie meczu. Odpowiedni model predykcyjny mógłby pomóc graczom podjąć decyzję o poddaniu meczu wcześniej, co potencjalnie **oszczędza czas gry** i unika niepotrzebnych strat.

**Role w drużynie:**
- **Top lane (Top)** – frontline, pojedynki 1v1.  
- **Mid lane (Mid)** – centralna rola, zadawanie obrażeń, kontrola mapy.  
- **Jungle** – poruszanie się po lesie, wsparcie drużyny, kontrola celów.  
- **ADC (Attack Damage Carry)** – główne źródło obrażeń fizycznych.  
- **Support** – ochrona sojuszników, inicjacja walk, kontrola wizji.

Na niskich rangach największy wpływ na rozgrywkę ma zazwyczaj rola **Jungle**, ponieważ gracze koncentrują się na przechwytywaniu dużych celów, które mają znaczący wpływ na dalszą część meczu. Zasadność tego założenia zostanie sprawdzona w analizie.

**Spodziewane rezultaty:**
- Skuteczność przewidywania wyników meczu ~80%.  
- Największy wpływ mają zabójstwa, złoto i efektywność gry junglera.

**Źródło danych i sposób pobrania:**  
Na potrzeby projektu dane zostały pobrane za pomocą **API Riot Games**, oficjalnego interfejsu pozwalającego na dostęp do statystyk graczy i meczów League of Legends. Proces pozyskiwania danych przebiegał w następujący sposób:  

1. **Query do dywizji Gold** – pobranie identyfikatorów graczy (PUUID) z wybranej dywizji.  
2. **Pobranie ostatniego losowego meczu** dla każdego gracza na podstawie PUUID (identyfikatora unikalnego dla gracza).  
3. **Pobranie danych do 15. minuty meczu** (`timeline15`) – czyli informacji o statystykach każdego gracza w kluczowej fazie wczesnej gry.  

Dzięki został uzyskany spójny zbiór danych, który pozwala analizować przewagę drużyny w pierwszych 15 minutach i budować modele predykcyjne przewidujące wynik meczu.



**Możliwości analizy w czasie rzeczywistym:**  
API Riot Games pozwala również na pobieranie danych **w trakcie trwania meczu**, co otwiera możliwość tworzenia modeli predykcyjnych działających w czasie rzeczywistym.  
Taki model mógłby służyć jako **wirtualny coach** – nie tylko dla pojedynczego gracza, jak robią narzędzia typu *Porofessor*, ale dla całej drużyny.  
Na przykład, analiza danych do 15. minuty mogłaby pomóc zidentyfikować **ogólne problemy w drużynie**, wskazać słabe punkty i zasugerować najlepsze decyzje strategiczne, co mogłoby skrócić czas gry i zwiększyć szanse na zwycięstwo.


""")


# Załaduj dane
data_path = "data/output/gold_full.csv"
df = pd.read_csv(data_path)

# Filtracja remake’ów
st.sidebar.header("Filtry")
remove_remakes = st.sidebar.checkbox("Usuń remake'i", True)
if remove_remakes:
    df = df[(df['gold_avg'] >= 1000) & (df['level_avg'] >= 3)]

# Wyodrębnij drużyny i posortuj
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

#st.write("Liczba wierszy po filtracji:", df.shape[0])
#st.write("Liczba meczów po połączeniu drużyn:", df_matches.shape[0])

#Podstawowe statystyki nowych cech
st.subheader("Podstawowe statystyki różnic drużyn")
st.dataframe(df_matches.describe().T)


st.markdown("""
- **gold_avg_diff** – różnica średniego złota między drużynami (team100 – team200). Złoto pozwala kupować przedmioty zwiększające siłę postaci, więc przewaga w złocie zwykle daje lepsze możliwości w walce.  
- **cs_avg_diff** – różnica średniej liczby zabitych jednostek (minionów/monsterów) między drużynami. Więcej CS = więcej złota i doświadczenia.  
- **jungle_cs_avg_diff** – różnica średniej liczby potworów zabitych przez junglera drużyny. Kontrola jungli wpływa na przewagę strategiczną i dostęp do celów mapy.  
- **level_avg_diff** – różnica średniego poziomu postaci między drużynami. Wyższy poziom daje lepsze umiejętności i większą siłę w walkach.  
- **xp_avg_diff** – różnica średniego doświadczenia (experience) zdobytego przez graczy. Wyższe XP pozwala szybciej zdobywać poziomy i umiejętności.  
- **total_damage_done_avg_diff** – różnica średniego zadawanych obrażeń (wszystkie źródła) drużyn. Pokazuje, która drużyna jest agresywniejsza i skuteczniejsza w zadawaniu obrażeń.  
- **total_damage_taken_avg_diff** – różnica średnich obrażeń otrzymanych przez drużyny. Może wskazywać, która drużyna jest bardziej odporna lub lepiej pozycjonuje swoich graczy.  
- **damage_to_champions_avg_diff** – różnica średnich obrażeń zadanych przeciwnym bohaterom (champions). Ważny wskaźnik skuteczności w walkach drużynowych.  
- **kills_avg_diff** – różnica średniej liczby zabójstw bohaterów przeciwnika. Bezpośrednio wpływa na przewagę w złocie i kontroli mapy.  
- **assists_avg_diff** – różnica średniej liczby asyst w zabójstwach. Pokazuje współpracę drużynową i skuteczność w wspieraniu sojuszników.  
- **towers_diff** – różnica liczby zniszczonych wież przez drużyny. Kontrolowanie wież daje przewagę na mapie i dostęp do wrogiego terytorium.  
- **dragons_diff** – różnica liczby smoków zabitych przez drużyny. Smoki dają trwałe bonusy, więc ich przewaga jest strategicznie istotna.  
- **first_blood_diff** – różnica, która drużyna zdobyła pierwszą krew (pierwsze zabójstwo). Pierwsze zabójstwo daje dodatkowe złoto i przewagę psychologiczną.  
- **first_tower_diff** – różnica, która drużyna zniszczyła pierwszą wieżę. Pierwsza wieża daje dodatkowe złoto i kontrolę mapy.  
- **first_dragon_diff** – różnica, która drużyna zdobyła pierwszego smoka. Pierwszy smok daje drużynie przewagę w buffach.  
- **win_team100** – zmienna celu, 1 jeśli drużyna 100 wygrała mecz, 0 jeśli przegrała.
""")
# Heatmapa korelacji
st.subheader("Mapa korelacji cech (różnice drużyn)")
numeric_cols = df_matches.select_dtypes(include=['int64', 'float64']).drop(columns=['win_team100'])
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.markdown("""
### Analiza korelacji cech

Na podstawie mapy korelacji widzimy, że niektóre cechy są silnie ze sobą powiązane, co jest naturalne w kontekście gry:

- **gold_avg_diff** silnie koreluje z **xp_avg_diff** (0.89) i **kills_avg_diff** (0.89), co pokazuje, że przewaga w złocie idzie w parze z przewagą w poziomach postaci oraz liczbie zabójstw.  
- **level_avg_diff** i **xp_avg_diff** mają bardzo wysoką korelację (0.94), co jest logiczne, bo wyższy poziom postaci wynika z większego doświadczenia zdobytego w grze.  
- **towers_diff** jest mocno skorelowane ujemnie z **gold_avg_diff** (-0.76) i **first_tower_diff** (0.87), co wskazuje, że pierwsza wieża daje znaczną przewagę w złocie i kontroli mapy.  
- **dragons_diff** i **first_dragon_diff** są bardzo silnie skorelowane (0.95), co pokazuje, że drużyna, która zdobywa pierwszego smoka, zdobywa ich więcej w ciągu gry, lub do 15 minuty większość drużyn zdobywa tylko jednego smoka.  
- **damage_to_champions_avg_diff** i **kills_avg_diff** mają wysoką korelację (0.71), co pokazuje, że drużyny z większą liczbą zabójstw zadają też więcej obrażeń bohaterom przeciwnika.  
- **assists_avg_diff** jest umiarkowanie skorelowane z **kills_avg_diff** (0.78), co pokazuje, że współpraca drużynowa przy zabójstwach ma znaczenie.  

Wnioski dla modelu:
- Niektóre cechy są mocno skorelowane (np. złoto, XP, level, kills). Z pewnością będą one miały duży wpływ na wynik meczu. 
- Cechy dotyczące pierwszych celów (first_blood_diff, first_tower_diff, first_dragon_diff) mają mniejsze korelacje z innymi zmiennymi, ale mogą mieć duży wpływ psychologiczny i strategiczny na wynik meczu.  
- Wysokie korelacje między **dragon_diff** a **first_dragon_diff** oraz **tower_diff** a **first_tower_diff** są sensowne. Silna korelacja oznacza, że zdobycie pierwszych celów może pozytywnie wpływać na dalszą rozgrywkę, natomiast sytuacje, gdy drużyna zdobywa pierwszy cel, lecz ma mniej celów do 15 minuty, potencjalnie świadczą o różnicach sytuacji między poszczególnymi liniami.
""")

# Podgląd danych
#st.subheader("Podgląd danych po połączeniu drużyn")
#n_rows = st.sidebar.slider("Liczba wierszy do podglądu:", min_value=5, max_value=50, value=10)
#st.dataframe(df_matches.head(n_rows))

# ========================================================================
# EKSPLORACYJNA ANALIZA DANYCH (EDA)
# ========================================================================
st.header("Eksploracyjna Analiza Danych (EDA)")

# Balans klas
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

# Analiza rozkładów kluczowych zmiennych
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

# Boxploty - porównanie cech w zależności od wyniku
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
### Wnioski z Analizy Eksploracyjnej (EDA)

#### 1. Balans Klas (Wynik Meczu)
* **Zbalansowany Zbiór:** Rozkład zwycięstw i porażek jest zbliżony do proporcji **50/50** (z lekkim wskazaniem na jedną ze stron). To kluczowa informacja, która sugeruje, że dane są reprezentatywne i nie wymagają stosowania technik takich jak oversampling czy undersampling przed przystąpieniem do modelowania.

#### 2. Rozkłady Kluczowych Zmiennych (Histogramy)
* **Symetria Rozkładów:** Większość cech różnicowych (gold, xp, kills, cs, damage) wykazuje **rozkład normalny (Gaussa)** wycentrowany wokół zera. 
* **Charakterystyka Rozgrywek:** Fakt, że większość obserwacji skupia się blisko zera, świadczy o tym, że większość meczów w zbiorze to starcia stosunkowo wyrównane. Skrajne przewagi (tzw. "stompy") stanowią mniejszość statystyczną.
* **Dyskretność Wież:** Różnica w zniszczonych wieżach (`towers_diff`) jako jedyna ma charakter skokowy, co wynika z natury tego obiektu w grze.

#### 3. Wpływ Cech na Wynik (Boxploty)
* **Ekonomia i Doświadczenie (Gold & XP):** To najsilniejsze predyktory zwycięstwa. Mediany dla wygranych i przegranych są wyraźnie odseparowane. Wygrana drużyna niemal zawsze utrzymuje dodatnią różnicę w złocie i doświadczeniu.
* **Zabójstwa (Kills) vs. Farma (CS):** Obie cechy silnie korelują z wynikiem, jednak różnica w CS (`cs_avg_diff`) wykazuje mniejszą liczbę wartości odstających w porównaniu do zabójstw. Sugeruje to, że stabilna przewaga w farmie jest bezpieczniejszym wskaźnikiem wygranej niż agresywne szukanie zabójstw.
* **Obrażenia (Damage to Champions):** Co ciekawe, mimo że wygrani zadają średnio więcej obrażeń, pudełka (interquartile range) w dużej mierze się pokrywają. Oznacza to, że same obrażenia nie są tak determinujące jak zdobyte złoto czy cele mapy.
* **Struktury (Towers):** Wyraźna separacja w `towers_diff` potwierdza, że niszczenie wież jest bezpośrednio powiązane z wynikiem meczu – mediana dla przegranych znajduje się poniżej zera, podczas gdy dla wygranych jest wyraźnie dodatnia.

#### 4. Podsumowanie dla Modelowania
* Cechy oparte na **zasobach (Gold, XP)** będą miały największą wagę w modelu.
* Występowanie wartości odstających (outliers) w statystykach zabójstw i obrażeń sugeruje, że model powinien być odporny na szum (np. algorytmy drzewiaste jak XGBoost czy LightGBM).
""")

# ========================================================================
#  PRZYGOTOWANIE DANYCH
# ========================================================================
st.header("Przygotowanie danych do modelowania")

# Wybór cech i zmiennej celu
X = df_matches.drop(columns=['matchId', 'win_team100'])
y = df_matches['win_team100']

st.write(f"**Liczba cech:** {X.shape[1]}")
st.write(f"**Liczba obserwacji:** {X.shape[0]}")
st.write(f"**Lista cech:** {list(X.columns)}")

# Podział na zbiór uczący i testowy
test_size = st.sidebar.slider("Rozmiar zbioru testowego (%):", min_value=10, max_value=40, value=20) / 100
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

st.write(f"**Zbiór uczący:** {X_train.shape[0]} obserwacji")
st.write(f"**Zbiór testowy:** {X_test.shape[0]} obserwacji")

# Standaryzacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Dane zostały wystandaryzowane (średnia=0, odchylenie standardowe=1)")

# 1️⃣4️⃣ Opcjonalnie: SMOTE (oversampling) jeśli klasy są niezbalansowane
#use_smote = st.sidebar.checkbox("Użyj SMOTE (oversampling)", False)
#if use_smote:
 #   smote = SMOTE(random_state=random_state)
 #   X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
 #   st.write(f" SMOTE zastosowany. Nowa liczba obserwacji w zbiorze uczącym: {X_train_scaled.shape[0]}")

# ========================================================================
# MODELOWANIE - UCZENIE MASZYNOWE
# ========================================================================
st.header("Modelowanie - Uczenie Maszynowe")

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
st.subheader("Regresja Logistyczna")

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
st.subheader("K-Nearest Neighbors (KNN)")

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
st.subheader("Drzewa Decyzyjne")

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
    st.markdown("""
Drzewo decyzyjne pokazuje, że kluczowym czynnikiem wpływającym na wynik meczu jest przewaga ekonomiczna drużyny. Różnica w zdobywanym złocie stanowi pierwszy i najważniejszy podział w modelu, co wskazuje na fundamentalne znaczenie ekonomii gry w prognozowaniu zwycięstwa lub porażki.
W sytuacjach niekorzystnej przewagi ekonomicznej model koncentruje się na różnicach w doświadczeniu oraz zadawanych obrażeniach. Gałęzie te najczęściej prowadzą do klasyfikacji porażki, co sugeruje, że brak złota trudno zrekompensować innymi statystykami.
Z kolei przy przewadze ekonomicznej istotną rolę odgrywa kontrola obiektów mapy oraz przewaga w doświadczeniu, które dodatkowo zwiększają prawdopodobieństwo wygranej. Model wskazuje, że zwycięstwa są efektem łącznej dominacji ekonomicznej i strategicznej, a uzyskane wyniki są spójne z logiką rozgrywki.
    """)

# ========================================================================
# MODEL 4: SUPPORT VECTOR MACHINE (SVM)
# ========================================================================
st.subheader("Support Vector Machine (SVM)")

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
st.header("Porównanie modeli")

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
st.markdown("""
Porównanie wyników pokazuje wyraźne różnice w skuteczności poszczególnych modeli klasyfikacyjnych. Model SVM charakteryzuje się najbardziej zrównoważonymi wynikami, osiągając wysoką skuteczność predykcji oraz dobre wartości miar precyzji, czułości i F1-score, co wskazuje na jego stabilność i dobrą zdolność generalizacji. Regresja logistyczna wyróżnia się bardzo dobrą zdolnością rozróżniania klas, co potwierdza wysoka wartość AUC-ROC, jednak jej skuteczność klasyfikacji jest nieco niższa w porównaniu do najlepszego modelu. Drzewo decyzyjne osiąga wysoką czułość, co oznacza dobrą identyfikację przypadków pozytywnych, lecz odbywa się to kosztem niższej precyzji. Model KNN wypada najsłabiej spośród porównywanych metod, osiągając niższe wartości większości metryk. Na podstawie uzyskanych wyników można stwierdzić, że SVM stanowi najlepszy kompromis pomiędzy skutecznością, stabilnością i jakością klasyfikacji.
""")
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
st.markdown("""
Analiza macierzy konfuzji pokazuje, że wszystkie modele poprawnie klasyfikują większość obserwacji, jednak różnią się strukturą popełnianych błędów. Regresja logistyczna charakteryzuje się stosunkowo zrównoważonym rozkładem błędów, co oznacza podobną skuteczność w identyfikacji obu klas. Model KNN częściej myli klasy, szczególnie w przypadku błędnej klasyfikacji porażek jako zwycięstw, co wskazuje na jego mniejszą stabilność. Drzewo decyzyjne wykazuje tendencję do lepszej identyfikacji zwycięstw kosztem większej liczby błędów dla klasy porażki, co potwierdza jego wysoką czułość. Model SVM osiąga najbardziej zrównoważone wyniki, charakteryzując się dużą liczbą poprawnych klasyfikacji przy relatywnie niskiej liczbie błędów dla obu klas. Uzyskane wyniki potwierdzają, że SVM najlepiej radzi sobie z rozróżnianiem klas przy zachowaniu stabilnej struktury błędów.
""")
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
st.markdown("""
Analiza krzywych ROC pokazuje, że wszystkie porównywane modele osiągają wyniki istotnie lepsze niż klasyfikator losowy, co potwierdza ich zdolność do skutecznego rozróżniania klas. Najlepsze właściwości separacyjne wykazują regresja logistyczna oraz SVM, których krzywe przez większość zakresu znajdują się najwyżej, co oznacza dobrą równowagę pomiędzy czułością a odsetkiem fałszywych alarmów. Drzewo decyzyjne osiąga nieco słabsze, lecz nadal stabilne rezultaty, natomiast model KNN wypada najsłabiej, szczególnie w obszarze niskich wartości fałszywie pozytywnych klasyfikacji. Wyniki wskazują, że modele liniowe oraz SVM charakteryzują się lepszą zdolnością generalizacji i bardziej stabilnym zachowaniem przy zmianie progu decyzyjnego.
""")
# ========================================================================
# INTERPRETOWALNOŚĆ - SHAP VALUES
# ========================================================================
st.header("Interpretowalność modelu - SHAP Values")

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
    st.markdown("""
    Analiza wartości SHAP dla najlepszego modelu pokazuje, że największy wpływ na predykcję wyniku meczu mają cechy związane z przewagą ekonomiczną oraz tempem rozwoju drużyny. Na pierwszym planie dominuje różnica w średnim złocie, której wysokie wartości jednoznacznie zwiększają prawdopodobieństwo zwycięstwa, natomiast niskie sprzyjają porażce. Istotną rolę odgrywają również różnice w kontroli kluczowych obiektów mapy oraz zdobywanym doświadczeniu, co potwierdza znaczenie aspektów makrogry i zarządzania zasobami.

Cechy związane z obrażeniami, zarówno zadawanymi, jak i otrzymywanymi, wpływają na predykcję w sposób bardziej zróżnicowany, wzmacniając lub osłabiając wynik w zależności od kontekstu pozostałych zmiennych. Z kolei statystyki o charakterze bardziej szczegółowym, takie jak liczba zabójstw, asyst czy pierwsza krew, mają relatywnie niewielki wpływ na końcową decyzję modelu.

Uzyskane wyniki wskazują, że model opiera swoje decyzje głównie na cechach odzwierciedlających długofalową przewagę drużyny, a nie na pojedynczych, incydentalnych zdarzeniach w trakcie meczu.
    
    """)
    
    # Bar plot (średnia wartość SHAP)
    st.subheader("Średnia wartość SHAP dla każdej cechy")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type='bar', show=False)
    st.pyplot(fig)
    st.markdown("""
Wykres średnich wartości SHAP pokazuje globalną istotność cech w najlepszym modelu, wskazując, które zmienne mają największy wpływ na podejmowane predykcje niezależnie od kierunku oddziaływania. Największe znaczenie mają cechy związane z przewagą ekonomiczną oraz kontrolą kluczowych elementów mapy, co potwierdza, że model w głównej mierze opiera swoje decyzje na długofalowej dominacji drużyny. Istotną rolę odgrywa również tempo rozwoju postaci, wyrażone poprzez różnice w doświadczeniu, które wzmacniają lub osłabiają przewagę ekonomiczną.

Cechy związane z obrażeniami mają umiarkowany wpływ na wynik predykcji, natomiast statystyki o bardziej szczegółowym charakterze, takie jak pojedyncze zdarzenia czy indywidualne akcje, charakteryzują się relatywnie niską istotnością globalną. Wyniki te wskazują, że model preferuje stabilne i systematyczne przewagi drużynowe zamiast krótkotrwałych, losowych zdarzeń.
""")
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
    st.markdown("""
Wykres SHAP Waterfall przedstawia lokalne wyjaśnienie pojedynczej predykcji modelu, pokazując, w jaki sposób poszczególne cechy wpływały na końcową decyzję. Predykcja zwycięstwa jest wynikiem skumulowanego wpływu kilku kluczowych zmiennych, z których największe znaczenie mają cechy związane z przewagą ekonomiczną oraz tempem rozwoju drużyny. Ich dodatni wpływ systematycznie przesuwa predykcję w stronę wygranej.

Cechy związane z kontrolą mapy oraz zadawanymi obrażeniami dodatkowo wzmacniają pozytywną decyzję modelu, natomiast zmienne o charakterze bardziej szczegółowym mają niewielki wpływ i jedynie korygują końcowy wynik. Negatywne oddziaływanie części cech nie jest wystarczające, aby zrównoważyć dominującą przewagę kluczowych czynników.
    """) 
    actual_label = "Wygrana" if y_test.iloc[sample_idx] == 1 else "Przegrana"
    predicted_label = "Wygrana" if results[best_model_name]['y_pred'][sample_idx] == 1 else "Przegrana"
    st.write(f"**Rzeczywista klasa:** {actual_label}")
    st.write(f"**Przewidywana klasa:** {predicted_label}")


# ========================================================================
# PODSUMOWANIE I WNIOSKI
# ========================================================================
st.header("Podsumowanie i Wnioski")

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

1. **Skuteczność predykcji**: Wszystkie modele osiągnęły wysoką dokładność (accuracy > 70%), co sugeruje, 
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

st.success("Projekt streamlit zakończony pomyślnie")

