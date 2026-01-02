"""
Interaktywna aplikacja EDA w Streamlit dla danych League of Legends.

Uruchomienie:
    streamlit run analysis/eda_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Konfiguracja strony
st.set_page_config(
    page_title="LoL Dataset EDA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style dla wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@st.cache_data
def load_data(filepath="data/output/gold_dataset.csv"):
    """Wczytuje dane z pliku CSV."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Plik {filepath} nie został znaleziony!")
        return None


def display_data_info(df):
    """Wyświetla podstawowe informacje o zbiorze danych."""
    st.header("1. Wczytanie danych")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liczba obserwacji", df.shape[0])
    with col2:
        st.metric("Liczba zmiennych", df.shape[1])
    with col3:
        st.metric("Rozmiar w pamięci", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.subheader("Nazwy kolumn i typy danych")
    
    # Informacje o typach danych
    dtypes_df = pd.DataFrame({
        'Kolumna': df.columns,
        'Typ': df.dtypes.values,
        'Unikalnych wartości': [df[col].nunique() for col in df.columns],
        'Braków danych': [df[col].isnull().sum() for col in df.columns]
    })
    st.dataframe(dtypes_df, use_container_width=True)
    
    st.subheader("Podgląd danych")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pierwsze 5 wierszy:**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Ostatnie 5 wierszy:**")
        st.dataframe(df.tail(), use_container_width=True)
    
    # Komentarz analityczny
    st.info("""
    **Obserwacje:**
    - Dataset zawiera dane z meczów League of Legends w formacie różnic między drużynami
    - Zmienne kończące się na `_diff` reprezentują różnicę statystyk między drużyną 100 a 200
    - Zmienna `win` to zmienna binarna (target) - czy drużyna wygrała (1) czy przegrała (0)
    - Dane dotyczą stanu gry na 15. minucie meczu
    """)


def analyze_data_types(df):
    """Analiza typów zmiennych."""
    st.header("2. Wstępna analiza danych")
    
    # Identyfikacja typów zmiennych
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Zmiennych numerycznych", len(numeric_cols))
        with st.expander("Pokaż listę"):
            st.write(numeric_cols)
    
    with col2:
        st.metric("Zmiennych kategorycznych", len(categorical_cols))
        with st.expander("Pokaż listę"):
            st.write(categorical_cols)
    
    return numeric_cols, categorical_cols


def analyze_missing_data(df):
    """Analiza braków danych."""
    st.subheader("Braki danych")
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Kolumna': missing.index,
        'Liczba braków': missing.values,
        'Procent braków': missing_percent.values
    }).sort_values('Liczba braków', ascending=False)
    
    missing_df = missing_df[missing_df['Liczba braków'] > 0]
    
    if missing_df.empty:
        st.success("Brak braków danych w zbiorze.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                missing_df,
                x='Kolumna',
                y='Procent braków',
                title='Procent braków danych w kolumnach',
                labels={'Procent braków': 'Procent braków (%)'},
                color='Procent braków',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(missing_df, use_container_width=True)
    
    return missing_df


def descriptive_statistics(df, numeric_cols):
    """Statystyki opisowe."""
    st.subheader("Statystyki opisowe")
    
    tab1, tab2 = st.tabs(["Zmienne numeryczne", "Zmienne kategoryczne"])
    
    with tab1:
        st.write("**Podstawowe statystyki zmiennych numerycznych:**")
        desc_stats = df[numeric_cols].describe().T
        desc_stats['skewness'] = df[numeric_cols].skew()
        desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
        
        st.dataframe(desc_stats.style.format("{:.2f}"), use_container_width=True)
        
        st.info("""
        **Interpretacja:**
        - **mean/median**: Średnia i mediana - porównaj je, aby wykryć skośność rozkładu
        - **std**: Odchylenie standardowe - wysoka wartość wskazuje dużą zmienność
        - **skewness**: Skośność - wartości bliskie 0 oznaczają rozkład symetryczny
        - **kurtosis**: Kurtoza - wysoka wartość wskazuje na "ciężkie ogony" rozkładu
        """)
    
    with tab2:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.write("**Statystyki zmiennych kategorycznych:**")
            
            for col in categorical_cols:
                with st.expander(f"{col}"):
                    value_counts = df[col].value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"Unikalnych wartości: **{df[col].nunique()}**")
                        st.dataframe(value_counts, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f'Rozkład kategorii: {col}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak zmiennych kategorycznych w zbiorze.")


def handle_missing_data(df, missing_df):
    """Uzupełnienie braków danych."""
    st.header("3. Analiza i uzupełnienie braków danych")
    
    if missing_df.empty:
        st.success("Brak braków danych do uzupełnienia.")
        return df.copy()
    
    st.write("**Strategia imputacji:**")
    
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    imputation_info = []
    
    # Uzupełnianie zmiennych numerycznych
    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
            imputation_info.append({
                'Kolumna': col,
                'Typ': 'Numeryczna',
                'Metoda': 'Mediana',
                'Wartość': f"{median_val:.2f}"
            })
    
    # Uzupełnianie zmiennych kategorycznych
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_val = df_imputed[col].mode()[0]
            df_imputed[col].fillna(mode_val, inplace=True)
            imputation_info.append({
                'Kolumna': col,
                'Typ': 'Kategoryczna',
                'Metoda': 'Moda',
                'Wartość': str(mode_val)
            })
    
    if imputation_info:
        st.dataframe(pd.DataFrame(imputation_info), use_container_width=True)
        st.success(f"Uzupełniono braki w {len(imputation_info)} kolumnach.")
    
    # Weryfikacja
    remaining_missing = df_imputed.isnull().sum().sum()
    st.metric("Pozostałe braki danych", remaining_missing)
    
    if remaining_missing == 0:
        st.success("Wszystkie braki danych zostały uzupełnione.")
    
    st.info("""
    **Uzasadnienie metody:**
    - **Mediana dla zmiennych numerycznych**: Odporna na wartości odstające, lepszy wybór niż średnia dla rozkładów skośnych
    - **Moda dla zmiennych kategorycznych**: Najczęstsza wartość w zbiorze, sensowne wypełnienie dla danych kategorycznych
    """)
    
    return df_imputed


def analyze_distributions(df, numeric_cols):
    """Analiza rozkładów zmiennych."""
    st.header("4. Analiza rozkładów zmiennych")
    
    st.subheader("Zmienne numeryczne")
    
    selected_var = st.selectbox(
        "Wybierz zmienną do analizy:",
        numeric_cols,
        key='dist_selector'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram z KDE
        fig, ax = plt.subplots(figsize=(10, 6))
        df[selected_var].hist(bins=30, alpha=0.7, edgecolor='black', ax=ax, density=True)
        df[selected_var].plot(kind='kde', ax=ax, linewidth=2, color='red')
        ax.set_title(f'Histogram i KDE: {selected_var}', fontsize=14, fontweight='bold')
        ax.set_xlabel(selected_var)
        ax.set_ylabel('Gęstość')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Boxplot
        fig = px.box(
            df,
            y=selected_var,
            title=f'Boxplot: {selected_var}',
            labels={selected_var: selected_var}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statystyki dla wybranej zmiennej
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Średnia", f"{df[selected_var].mean():.2f}")
    with col2:
        st.metric("Mediana", f"{df[selected_var].median():.2f}")
    with col3:
        st.metric("Odch. std", f"{df[selected_var].std():.2f}")
    with col4:
        st.metric("Skośność", f"{df[selected_var].skew():.2f}")
    
    st.info(f"""
    **Interpretacja dla {selected_var}:**
    - Rozkład {'symetryczny' if abs(df[selected_var].skew()) < 0.5 else 'skośny'}
    - {'Brak wyraźnych' if abs(df[selected_var].skew()) < 0.5 else 'Występują'} wartości odstające widoczne na boxplocie
    - Zmienność: {'niska' if df[selected_var].std() / abs(df[selected_var].mean()) < 0.5 else 'wysoka'}
    """)
    
    # Rozkłady wszystkich zmiennych - heatmap
    st.subheader("Przegląd wszystkich rozkładów")
    
    if st.checkbox("Pokaż macierz rozkładów wszystkich zmiennych"):
        fig = make_subplots(
            rows=(len(numeric_cols) + 3) // 4,
            cols=4,
            subplot_titles=numeric_cols[:16]  # Ograniczenie do 16 zmiennych
        )
        
        for idx, col in enumerate(numeric_cols[:16]):
            row = idx // 4 + 1
            col_num = idx % 4 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row,
                col=col_num
            )
        
        fig.update_layout(height=300 * ((len(numeric_cols[:16]) + 3) // 4), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def analyze_categorical(df):
    """Analiza zmiennych kategorycznych."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.info("Brak zmiennych kategorycznych do analizy.")
        return
    
    st.subheader("Zmienne kategoryczne")
    
    selected_cat = st.selectbox(
        "Wybierz zmienną kategoryczną:",
        categorical_cols,
        key='cat_selector'
    )
    
    value_counts = df[selected_cat].value_counts()
    
    # Ogranicz do top 20 kategorii dla czytelności
    top_n = 20
    if len(value_counts) > top_n:
        st.warning(f"Zmienna ma {len(value_counts)} unikalnych wartości. Pokazuję tylko top {top_n}.")
        value_counts_display = value_counts.head(top_n)
    else:
        value_counts_display = value_counts
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**Częstości (top {min(top_n, len(value_counts))}):**")
        st.dataframe(value_counts_display, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=value_counts_display.index,
            y=value_counts_display.values,
            title=f'Rozkład kategorii: {selected_cat} (top {min(top_n, len(value_counts))})',
            labels={'x': selected_cat, 'y': 'Liczność'},
            color=value_counts_display.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis={'tickangle': -45})
        st.plotly_chart(fig, use_container_width=True)
    
    # Dodatkowa informacja
    st.info(f"""
    **Statystyki:**
    - Unikalnych wartości: {len(value_counts)}
    - Najczęstsza wartość: {value_counts.index[0]} ({value_counts.values[0]} wystąpień)
    - Najrzadsza wartość: {value_counts.index[-1]} ({value_counts.values[-1]} wystąpień)
    """)


def detect_outliers(df, numeric_cols):
    """Wykrywanie wartości odstających metodą IQR."""
    st.header("5. Analiza wartości odstających")
    
    st.write("""
    Wartości odstające wykrywane są metodą **IQR (Interquartile Range)**:
    - Outlier jeśli: wartość < Q1 - 1.5*IQR lub wartość > Q3 + 1.5*IQR
    """)
    
    outlier_stats = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(df)) * 100
        
        outlier_stats.append({
            'Zmienna': col,
            'Liczba outlierów': outlier_count,
            'Procent': f"{outlier_percent:.2f}%",
            'Dolna granica': f"{lower_bound:.2f}",
            'Górna granica': f"{upper_bound:.2f}"
        })
    
    outlier_df = pd.DataFrame(outlier_stats).sort_values('Liczba outlierów', ascending=False)
    
    st.dataframe(outlier_df, use_container_width=True)
    
    # Wizualizacja zmiennej z największą liczbą outlierów
    top_outlier_col = outlier_df.iloc[0]['Zmienna']
    
    st.subheader(f"Wizualizacja outlierów dla: {top_outlier_col}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df,
            y=top_outlier_col,
            title=f'Boxplot z outlierami: {top_outlier_col}',
            points='outliers'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x=top_outlier_col,
            title=f'Histogram: {top_outlier_col}',
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.warning("""
    **Uwaga dotycząca outlierów:**
    - W danych z gier outliers mogą być **naturalne** (np. bardzo jednostronne mecze)
    - **Nie usuwamy ich automatycznie** - mogą zawierać ważne informacje
    - Przy modelowaniu ML warto przetestować modele z i bez outlierów
    - Modele oparte na drzewach (Random Forest, XGBoost) są odporne na outliery
    """)


def correlation_analysis(df, numeric_cols):
    """Analiza korelacji między zmiennymi."""
    st.header("6. Analiza zależności między zmiennymi")
    
    st.subheader("Macierz korelacji (Pearson)")
    
    # Oblicz korelację
    corr_matrix = df[numeric_cols].corr()
    
    # Heatmapa korelacji
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Macierz korelacji zmiennych numerycznych', fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig)
    plt.close()
    
    # Top korelacje
    st.subheader("Najsilniejsze korelacje")
    
    # Przekształć macierz korelacji w listę par
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Zmienna 1': corr_matrix.columns[i],
                'Zmienna 2': corr_matrix.columns[j],
                'Korelacja': corr_matrix.iloc[i, j]
            })
    
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Korelacja', key=abs, ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**10 najsilniejszych dodatnich korelacji:**")
        st.dataframe(
            corr_pairs_df[corr_pairs_df['Korelacja'] > 0].head(10).style.format({'Korelacja': '{:.3f}'}),
            use_container_width=True
        )
    
    with col2:
        st.write("**10 najsilniejszych ujemnych korelacji:**")
        st.dataframe(
            corr_pairs_df[corr_pairs_df['Korelacja'] < 0].head(10).style.format({'Korelacja': '{:.3f}'}),
            use_container_width=True
        )
    
    st.info("""
    **Interpretacja korelacji:**
    - **|r| > 0.7**: Silna korelacja - zmienne są ze sobą mocno związane
    - **0.3 < |r| < 0.7**: Umiarkowana korelacja
    - **|r| < 0.3**: Słaba korelacja
    - Wysokie korelacje między predyktorami mogą wskazywać na **multikolinearność**
    """)
    
    # Scatter plots interaktywne
    st.subheader("Interaktywna analiza par zmiennych")
    
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Wybierz zmienną X:", numeric_cols, key='scatter_x')
    with col2:
        var_y = st.selectbox("Wybierz zmienną Y:", numeric_cols, index=1, key='scatter_y')
    
    # Scatter plot z linią trendu
    fig = px.scatter(
        df,
        x=var_x,
        y=var_y,
        title=f'Zależność: {var_x} vs {var_y}',
        trendline='ols',
        color='win' if 'win' in df.columns else None,
        labels={'win': 'Wygrana (0/1)'},
        opacity=0.6
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Korelacja dla wybranej pary
    correlation = df[var_x].corr(df[var_y])
    st.metric(f"Korelacja Pearsona między {var_x} i {var_y}", f"{correlation:.3f}")


def target_analysis(df):
    """Analiza zmiennej docelowej i jej zależności z predyktorami."""
    st.header("7. Analiza zmiennej docelowej (Target)")
    
    if 'win' not in df.columns:
        st.warning("Brak zmiennej 'win' w zbiorze danych.")
        return
    
    st.subheader("Rozkład zmiennej docelowej (win)")
    
    col1, col2, col3 = st.columns(3)
    
    win_counts = df['win'].value_counts()
    
    with col1:
        st.metric("Wygrane (1)", win_counts.get(1, 0))
    with col2:
        st.metric("Przegrane (0)", win_counts.get(0, 0))
    with col3:
        balance = (win_counts.get(1, 0) / len(df)) * 100
        st.metric("Balance (%)", f"{balance:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=win_counts.values,
            names=['Przegrana', 'Wygrana'],
            title='Rozkład wyników meczów',
            color_discrete_sequence=['#EF553B', '#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=['Przegrana', 'Wygrana'],
            y=win_counts.values,
            title='Liczność klas',
            labels={'x': 'Wynik', 'y': 'Liczba obserwacji'},
            color=win_counts.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Balans klas
    if abs(balance - 50) < 10:
        st.success("Klasy są dobrze zbalansowane.")
    else:
        st.warning(f"Niezbalansowanie klas: {balance:.1f}% / {100-balance:.1f}%")
    
    # Korelacja predyktorów z target
    st.subheader("Korelacja zmiennych z wynikiem meczu (win)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'win' in numeric_cols:
        numeric_cols.remove('win')
    
    correlations_with_target = df[numeric_cols + ['win']].corr()['win'].drop('win').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 dodatnich korelacji z win:**")
        top_positive = correlations_with_target.head(10)
        fig = px.bar(
            x=top_positive.values,
            y=top_positive.index,
            orientation='h',
            title='Najsilniejsze dodatnie korelacje',
            labels={'x': 'Korelacja', 'y': 'Zmienna'},
            color=top_positive.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Top 10 ujemnych korelacji z win:**")
        top_negative = correlations_with_target.tail(10)
        fig = px.bar(
            x=top_negative.values,
            y=top_negative.index,
            orientation='h',
            title='Najsilniejsze ujemne korelacje',
            labels={'x': 'Korelacja', 'y': 'Zmienna'},
            color=top_negative.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Wnioski dla modelowania ML:**
    - Zmienne z wysoką korelacją z `win` będą najprawdopodobniej najważniejszymi predyktorami
    - Zmienne z `_diff` (różnice między drużynami) powinny być szczególnie istotne
    - `kills_diff`, `towers_diff`, `gold_diff` prawdopodobnie będą kluczowe dla modeli
    - Warto przetestować również interakcje między zmiennymi
    """)
    
    # Box plots dla top zmiennych
    st.subheader("📦 Rozkłady top zmiennych względem wyniku meczu")
    
    top_vars = correlations_with_target.abs().sort_values(ascending=False).head(6).index.tolist()
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=top_vars
    )
    
    for idx, var in enumerate(top_vars):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        for win_val in [0, 1]:
            fig.add_trace(
                go.Box(
                    y=df[df['win'] == win_val][var],
                    name=f'Win={win_val}',
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Główna funkcja aplikacji."""
    
    # Tytuł aplikacji
    st.title("League of Legends - Eksploracyjna Analiza Danych")
    
    # ⬇️ SEKCJA INFORMACYJNA - MOŻESZ TU EDYTOWAĆ ⬇️
    st.markdown("""
    ### O projekcie
    Aplikacja przeprowadza kompleksową **eksploracyjną analizę danych** (EDA) meczów League of Legends.
    
    **Cel projektu:**
    - Zrozumienie kluczowych czynników wpływających na wynik meczu
    - Przygotowanie danych do modelowania Machine Learning
    - Identyfikacja najważniejszych statystyk w pierwszych 15 minutach gry
    
    **Autor:** Kamil Ładyga, Miłosz Polinceusz
                
    **Data:** Styczeń 2026
    """)
    
    st.markdown("---")
    
    # Sidebar z nawigacją
    with st.sidebar:
        st.header("Konfiguracja")
        
        data_path = st.text_input(
            "Ścieżka do pliku CSV:",
            value="data/output/gold_dataset.csv"
        )
        
        st.markdown("---")
        st.header("Nawigacja")
        st.markdown("""
        1. Wczytanie danych
        2. Wstępna analiza
        3. Braki danych
        4. Rozkłady zmiennych
        5. Wartości odstające
        6. Korelacje
        7. Analiza targetu
        """)
        
        st.markdown("---")
        st.info("""
        **O aplikacji:**
        
        Interaktywna analiza EDA datasetu z meczów League of Legends.
        
        Dane reprezentują stan gry na 15. minucie meczu.
        """)
    
    # Wczytaj dane
    df = load_data(data_path)
    
    if df is None:
        st.stop()
    
    # Sekcje EDA
    display_data_info(df)
    
    st.markdown("---")
    numeric_cols, categorical_cols = analyze_data_types(df)
    
    missing_df = analyze_missing_data(df)
    descriptive_statistics(df, numeric_cols)
    
    st.markdown("---")
    df_clean = handle_missing_data(df, missing_df)
    
    st.markdown("---")
    analyze_distributions(df_clean, numeric_cols)
    analyze_categorical(df_clean)
    
    st.markdown("---")
    detect_outliers(df_clean, numeric_cols)
    
    st.markdown("---")
    correlation_analysis(df_clean, numeric_cols)
    
    st.markdown("---")
    target_analysis(df_clean)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Utworzono przy użyciu Streamlit | Dataset: League of Legends Ranked Games</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
