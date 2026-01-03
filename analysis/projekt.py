import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

