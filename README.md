Projekt – Analiza danych i uczenie maszynowe

Projekt został wykonany w ramach projektu zaliczeniowego na studiach.
Celem pracy jest analiza danych oraz budowa modeli uczenia maszynowego do przewidywania wyniku meczu League of Legends (wygrana/przegrana) na podstawie danych z pierwszych 15 minut gry.

Zakres projektu

Projekt spełnia następujące wymagania:

wstęp z określeniem celu pracy,

EDA – eksploracyjna analiza danych (statystyki, wizualizacje, balans klas, wpływ zmiennych),

przygotowanie danych (filtracja, standaryzacja, opcjonalny oversampling SMOTE),

podział danych na zbiór uczący i testowy,

zastosowanie co najmniej 3 metod uczenia maszynowego (m.in. SVM i drzewa decyzyjne),

porównanie jakości modeli (Accuracy, Precision, Recall, F1, AUC),

analiza interpretowalności modelu z wykorzystaniem SHAP,

podsumowanie oraz wnioski końcowe.

Dodatkowo zastosowano:

walidację krzyżową,

optymalizację hiperparametrów (GridSearchCV),

rozbudowaną wizualizację wyników.

Uruchomienie projektu

Projekt uruchamiany jest jako aplikacja Streamlit:

streamlit run analysis/project.py

Technologie

Python

Streamlit

pandas, numpy, matplotlib, seaborn

scikit-learn

imbalanced-learn (SMOTE)

SHAP


