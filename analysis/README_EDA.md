# EDA Streamlit App - README

## 🚀 Uruchomienie aplikacji

```bash
streamlit run analysis/eda_app.py
```

Lub z parametrem ścieżki do pliku:

```bash
streamlit run analysis/eda_app.py -- --data_path data/output/gold_full.csv
```

## 📋 Funkcjonalności

### 1️⃣ Wczytanie danych
- Liczba obserwacji i zmiennych
- Typy danych kolumn
- Podgląd head/tail

### 2️⃣ Wstępna analiza
- Identyfikacja typów zmiennych (numeryczne/kategoryczne)
- Liczba i procent braków danych
- Statystyki opisowe (mean, median, std, quartiles, skewness, kurtosis)

### 3️⃣ Braki danych
- Wizualizacja braków (wykres słupkowy)
- Automatyczna imputacja:
  - Zmienne numeryczne → mediana
  - Zmienne kategoryczne → moda
- Raport z uzupełnionych kolumn

### 4️⃣ Rozkłady zmiennych
- **Numeryczne**: histogramy + KDE, boxploty
- **Kategoryczne**: wykresy słupkowe częstości
- Interaktywny wybór zmiennej
- Statystyki dla wybranej zmiennej

### 5️⃣ Wartości odstające
- Detekcja metodą IQR
- Tabela z liczbą outlierów
- Wizualizacja boxplotami
- Rekomendacje (nie usuwamy automatycznie)

### 6️⃣ Korelacje
- Macierz korelacji (heatmapa)
- Top 10 najsilniejszych korelacji (+/-)
- Interaktywne scatter ploty par zmiennych
- Linie trendu (OLS)

### 7️⃣ Analiza targetu (win)
- Rozkład win/lose
- Balans klas
- Top korelacje z targetem
- Boxploty zmiennych względem wyniku

### 8️⃣ Rekomendacje ML
- Mocne strony datasetu
- Potencjalne wyzwania
- Sugerowane modele
- Pipeline preprocessingu
- Metryki ewaluacji

## 📦 Zależności

Wszystkie wymagane biblioteki są już w `requirements.txt`:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

## 🎨 Cechy aplikacji

✅ **Interaktywność**: Wybór zmiennych z dropdown, dynamiczne wykresy  
✅ **Wizualizacje**: Plotly (interaktywne) + Matplotlib/Seaborn  
✅ **Komentarze analityczne**: Interpretacja przy każdej sekcji  
✅ **Przygotowanie do ML**: Rekomendacje i best practices  
✅ **Responsywność**: Layout z kolumnami, clean design  

## 📊 Struktura danych

Aplikacja oczekuje pliku CSV z następującymi kolumnami:
- `win` - zmienna target (0/1)
- `*_diff` - różnice statystyk między drużynami
- Zmienne numeryczne: gold, cs, xp, damage, vision, kills, etc.

## 🔄 Workflow

1. **EDA** ← Jesteś tutaj
2. Podział train/test
3. Feature engineering
4. Trenowanie modeli ML (min. 3)
5. Ewaluacja i porównanie
6. Interpretowalność (SHAP, feature importance)

## 💡 Tips

- Użyj `--clear_cache` w sidebarie jeśli dane się zmieniły
- Eksportuj oczyszczone dane po imputacji
- Zwróć uwagę na multikolinearność przed ML
- Standaryzuj dane przed modelami liniowymi

## 🐛 Troubleshooting

**Błąd: "File not found"**
- Sprawdź ścieżkę w sidebarze
- Domyślnie: `data/output/gold_dataset.csv`

**Wolne ładowanie**
- Streamlit cache'uje dane
- Pierwsze uruchomienie może trwać dłużej

**Błąd importu plotly**
- `pip install plotly`

---

**Autor**: Generated for LoL ML Project  
**Stack**: Python + Streamlit + Plotly
