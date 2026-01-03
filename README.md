# Projekt – Analiza danych i uczenie maszynowe

Projekt został wykonany w ramach projektu zaliczeniowego na studiach.
Celem pracy jest analiza danych oraz budowa modeli uczenia maszynowego do przewidywania wyniku meczu League of Legends (wygrana/przegrana) na podstawie danych z pierwszych 15 minut gry w dywizji **Gold**.

## Kontekst projektu

**League of Legends (LoL)** to gra typu MOBA, w której rywalizują dwie pięcioosobowe drużyny. Projekt skupia się na analizie meczów w dywizji **Gold** (średni poziom rozgrywek), gdzie:
- Umiejętności strategiczne (makro) mają większe znaczenie niż indywidualna kontrola postaci (mikro)
- W 15. minucie gracze otrzymują możliwość poddania meczu
- Analiza danych do tego momentu pozwala oszacować szanse na wygraną i podjąć świadomą decyzję

**Źródło danych**: API Riot Games - oficjalny interfejs pozwalający na pobieranie statystyk meczów i graczy.

## Zakres projektu

Projekt spełnia wszystkie wymagania projektu zaliczeniowego:

1. **Wstęp z określeniem celu pracy** – szczegółowy opis gry, kontekstu dywizji Gold i celu projektu

2. **EDA – eksploracyjna analiza danych**:
   - Podstawowe statystyki i rozkłady zmiennych
   - Wizualizacje zależności między cechami
   - Analiza balansu klas
   - Badanie korelacji i wpływu zmiennych na wynik

3. **Przygotowanie danych**:
   - Filtracja i czyszczenie danych
   - Standaryzacja cech (StandardScaler)
   - Opcjonalne balansowanie klas (SMOTE)
   - Podział na zbiór uczący i testowy

4. **Zastosowanie metod uczenia maszynowego** (minimum 3):
   - **Support Vector Machine (SVM)** ✓
   - **Drzewa Decyzyjne (Decision Trees)** ✓
   - Regresja Logistyczna
   - K-Nearest Neighbors (KNN)

5. **Porównanie jakości modeli** z wykorzystaniem metryk:
   - Accuracy, Precision, Recall, F1-Score
   - AUC (Area Under Curve)
   - Confusion Matrix
   - ROC Curve

6. **Analiza interpretowalności modelu**:
   - Wykorzystanie **SHAP (SHapley Additive exPlanations)**
   - Identyfikacja najważniejszych cech
   - Wizualizacja wpływu zmiennych na predykcje

7. **Podsumowanie oraz wnioski końcowe**:
   - Ocena skuteczności modeli
   - Wnioski strategiczne dla graczy
   - Ograniczenia analizy i potencjalne rozszerzenia

## Dodatkowe elementy projektu

Projekt wykracza poza minimalne wymagania poprzez zastosowanie:

- **Walidacja krzyżowa (cross-validation)** – ocena stabilności modeli
- **Optymalizacja hiperparametrów (GridSearchCV)** – dostrojenie parametrów modeli
- **Rozbudowana wizualizacja wyników** – wykresy, heatmapy, wykresy SHAP
- **Interaktywna aplikacja Streamlit** – przejrzysta prezentacja wyników

## Uruchomienie projektu

Projekt uruchamiany jest jako aplikacja Streamlit:

```bash
streamlit run analysis/projekt.py
```

## Technologie

- **Python** – język programowania
- **Streamlit** – interfejs webowy do prezentacji wyników
- **pandas, numpy** – przetwarzanie i analiza danych
- **matplotlib, seaborn** – wizualizacja danych
- **scikit-learn** – modele uczenia maszynowego, metryki, preprocessing
- **imbalanced-learn (SMOTE)** – balansowanie klas
- **SHAP** – interpretowalność modeli

## Struktura projektu

```
├── analysis/           # Główna aplikacja i analiza
│   ├── projekt.py      # Główny plik projektu (Streamlit)
│   └── ...
├── api/                # Pobieranie danych z Riot API
├── data/               # Dane wejściowe i wyjściowe
│   └── output/
│       └── gold_full.csv
├── features/           # Feature engineering
├── stats/              # Statystyki i wykresy
└── requirements.txt    # Zależności projektu
```

## Wyniki

- Wszystkie modele osiągnęły **dokładność > 70%**
- Najważniejsze cechy: różnica w złocie, zabójstwach, XP i kontroli wież
- Model potwierdza, że dane z pierwszych 15 minut są wystarczające do przewidywania wyniku meczu
- Praktyczne zastosowanie: wsparcie decyzji o poddaniu meczu w 15. minucie


