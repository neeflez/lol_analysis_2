"""
Wizualizacja danych z plików CSV wygenerowanych przez counter_stats.py
Tworzy różne wykresy do analizy champion statistics, roles, win rates, counters itp.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_data(data_dir: str):
    """Wczytuje pliki CSV z katalogu"""
    roles_path = os.path.join(data_dir, "champion_roles.csv")
    counters_path = os.path.join(data_dir, "counters.csv")
    matchups_path = os.path.join(data_dir, "matchups_extra.csv")
    
    df_roles = pd.read_csv(roles_path) if os.path.exists(roles_path) else None
    df_counters = pd.read_csv(counters_path) if os.path.exists(counters_path) else None
    df_matchups = pd.read_csv(matchups_path) if os.path.exists(matchups_path) else None
    
    return df_roles, df_counters, df_matchups


def plot_top_champions_by_winrate(df_roles, output_dir, top_n=20):
    """Wykres: Top N championów według win rate"""
    if df_roles is None or df_roles.empty:
        print("Brak danych champion_roles")
        return
    
    # Filtrujemy tylko wiersze z win_rate
    df_valid = df_roles[df_roles['win_rate_pct'].notna()].copy()
    df_top = df_valid.nlargest(top_n, 'win_rate_pct')
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(df_top['champion'] + ' (' + df_top['role'] + ')', 
                    df_top['win_rate_pct'], 
                    color=sns.color_palette("viridis", len(df_top)))
    
    plt.xlabel('Win Rate (%)')
    plt.title(f'Top {top_n} Championów według Win Rate (wszystkie role)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Dodaj wartości na slupkach
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_top_winrate.png'), dpi=150)
    print(f"✓ Zapisano: 1_top_winrate.png")
    plt.close()


def plot_top_champions_by_pickrate(df_roles, output_dir, top_n=20):
    """Wykres: Top N championów według pick rate"""
    if df_roles is None or df_roles.empty:
        return
    
    df_valid = df_roles[df_roles['pick_rate_pct'].notna()].copy()
    df_top = df_valid.nlargest(top_n, 'pick_rate_pct')
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(df_top['champion'] + ' (' + df_top['role'] + ')', 
                    df_top['pick_rate_pct'],
                    color=sns.color_palette("mako", len(df_top)))
    
    plt.xlabel('Pick Rate (%)')
    plt.title(f'Top {top_n} Championów według Pick Rate (wszystkie role)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_top_pickrate.png'), dpi=150)
    print(f"✓ Zapisano: 2_top_pickrate.png")
    plt.close()


def plot_winrate_vs_pickrate_scatter(df_roles, output_dir):
    """Scatter plot: Win Rate vs Pick Rate"""
    if df_roles is None or df_roles.empty:
        return
    
    df_valid = df_roles[(df_roles['win_rate_pct'].notna()) & (df_roles['pick_rate_pct'].notna())].copy()
    
    plt.figure(figsize=(14, 10))
    
    # Różne kolory dla różnych ról
    roles = df_valid['role'].unique()
    colors = sns.color_palette("Set2", len(roles))
    
    for role, color in zip(roles, colors):
        df_role = df_valid[df_valid['role'] == role]
        plt.scatter(df_role['pick_rate_pct'], df_role['win_rate_pct'], 
                   label=role, alpha=0.6, s=100, color=color, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Pick Rate (%)', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.title('Win Rate vs Pick Rate (wszystkie championi i role)', fontsize=14, fontweight='bold')
    plt.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Linie referencyjne
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% WR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_winrate_vs_pickrate.png'), dpi=150)
    print(f"✓ Zapisano: 3_winrate_vs_pickrate.png")
    plt.close()


def plot_role_distribution(df_roles, output_dir):
    """Wykres: Rozkład championów per rola"""
    if df_roles is None or df_roles.empty:
        return
    
    role_counts = df_roles['role'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Wykres kołowy
    axes[0].pie(role_counts.values, labels=role_counts.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("pastel"))
    axes[0].set_title('Rozkład Championów według Ról', fontsize=12, fontweight='bold')
    
    # Wykres słupkowy
    axes[1].bar(role_counts.index, role_counts.values, color=sns.color_palette("pastel"))
    axes[1].set_xlabel('Rola')
    axes[1].set_ylabel('Liczba Championów')
    axes[1].set_title('Liczba Championów per Rola', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Dodaj wartości na słupkach
    for i, (role, count) in enumerate(role_counts.items()):
        axes[1].text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_role_distribution.png'), dpi=150)
    print(f"✓ Zapisano: 4_role_distribution.png")
    plt.close()


def plot_avg_stats_by_role(df_roles, output_dir):
    """Wykres: Średni Win Rate i Pick Rate per rola"""
    if df_roles is None or df_roles.empty:
        return
    
    df_valid = df_roles[(df_roles['win_rate_pct'].notna()) & (df_roles['pick_rate_pct'].notna())].copy()
    role_stats = df_valid.groupby('role').agg({
        'win_rate_pct': 'mean',
        'pick_rate_pct': 'mean'
    }).round(2)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Win Rate per rola
    axes[0].bar(role_stats.index, role_stats['win_rate_pct'], color=sns.color_palette("rocket", len(role_stats)))
    axes[0].set_ylabel('Średni Win Rate (%)')
    axes[0].set_title('Średni Win Rate według Ról', fontsize=12, fontweight='bold')
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, (role, wr) in enumerate(role_stats['win_rate_pct'].items()):
        axes[0].text(i, wr + 0.3, f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pick Rate per rola
    axes[1].bar(role_stats.index, role_stats['pick_rate_pct'], color=sns.color_palette("crest", len(role_stats)))
    axes[1].set_ylabel('Średni Pick Rate (%)')
    axes[1].set_title('Średni Pick Rate według Ról', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    for i, (role, pr) in enumerate(role_stats['pick_rate_pct'].items()):
        axes[1].text(i, pr + 0.1, f'{pr:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_avg_stats_by_role.png'), dpi=150)
    print(f"✓ Zapisano: 5_avg_stats_by_role.png")
    plt.close()


def plot_counter_ratings_distribution(df_counters, output_dir):
    """Wykres: Rozkład counter ratings"""
    if df_counters is None or df_counters.empty:
        return
    
    df_valid = df_counters[df_counters['counter_rating'].notna()].copy()
    
    plt.figure(figsize=(14, 6))
    plt.hist(df_valid['counter_rating'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Counter Rating')
    plt.ylabel('Częstotliwość')
    plt.title('Rozkład Counter Ratings', fontsize=14, fontweight='bold')
    plt.axvline(df_valid['counter_rating'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Średnia: {df_valid["counter_rating"].mean():.2f}')
    plt.axvline(df_valid['counter_rating'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f'Mediana: {df_valid["counter_rating"].median():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_counter_ratings_dist.png'), dpi=150)
    print(f"✓ Zapisano: 6_counter_ratings_dist.png")
    plt.close()


def plot_strongest_counters(df_counters, output_dir, top_n=20):
    """Wykres: Najsilniejsze counterowania (najwyższy counter rating)"""
    if df_counters is None or df_counters.empty:
        return
    
    df_valid = df_counters[df_counters['counter_rating'].notna()].copy()
    df_top = df_valid.nlargest(top_n, 'counter_rating')
    
    # Tworzymy etykiety
    labels = df_top.apply(lambda row: f"{row['champion']} ({row['role']}) vs {row['counter']}", axis=1)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(labels, df_top['counter_rating'], color=sns.color_palette("flare", len(df_top)))
    
    plt.xlabel('Counter Rating')
    plt.title(f'Top {top_n} Najsilniejszych Counterów', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_strongest_counters.png'), dpi=150)
    print(f"✓ Zapisano: 7_strongest_counters.png")
    plt.close()


def plot_most_countered_champions(df_counters, output_dir, top_n=15):
    """Wykres: Championi, którzy są najczęściej counterowani"""
    if df_counters is None or df_counters.empty:
        return
    
    # Zliczamy ile razy dany champion jest counterowany
    counter_counts = df_counters.groupby('champion').size().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(counter_counts.index, counter_counts.values, 
                    color=sns.color_palette("coolwarm", len(counter_counts)))
    
    plt.xlabel('Liczba Counterów')
    plt.title(f'Top {top_n} Najczęściej Counterowanych Championów', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                int(width), 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_most_countered.png'), dpi=150)
    print(f"✓ Zapisano: 8_most_countered.png")
    plt.close()


def plot_champion_counter_network(df_counters, output_dir, top_n=15):
    """Wykres: Top countery dla wybranych championów - wizualizacja relacji"""
    if df_counters is None or df_counters.empty:
        return
    
    # Wybierz top championów według liczby counterów
    top_champs = df_counters.groupby('champion').size().nlargest(top_n).index
    df_subset = df_counters[df_counters['champion'].isin(top_champs)].copy()
    
    # Grupuj countery dla każdego championa (top 3 countery)
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for idx, champ in enumerate(top_champs[:15]):
        ax = axes[idx]
        champ_data = df_subset[df_subset['champion'] == champ].nlargest(5, 'counter_rating')
        
        if not champ_data.empty:
            colors = sns.color_palette("Reds_r", len(champ_data))
            bars = ax.barh(champ_data['counter'], champ_data['counter_rating'], color=colors)
            ax.set_xlabel('Counter Rating', fontsize=9)
            ax.set_title(f'{champ}', fontsize=10, fontweight='bold')
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            
            # Dodaj wartości
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}', va='center', fontsize=7)
    
    # Ukryj puste subploty
    for idx in range(len(top_champs), 15):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top 5 Counterów dla {top_n} Najczęściej Counterowanych Championów', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_dir, '5_champion_counter_network.png'), dpi=150)
    print(f"✓ Zapisano: 5_champion_counter_network.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Wizualizacja danych z counter_stats.py")
    parser.add_argument("--data_dir", default="out", help="Katalog z plikami CSV")
    parser.add_argument("--output_dir", default="plots", help="Katalog wyjściowy dla wykresów")
    args = parser.parse_args()
    
    # Utwórz katalog wyjściowy
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("WIZUALIZACJA DANYCH LEAGUE OF LEGENDS")
    print("=" * 60)
    print(f"\nWczytywanie danych z: {args.data_dir}")
    
    # Wczytaj dane
    df_roles, df_counters, df_matchups = load_data(args.data_dir)
    
    if df_roles is not None:
        print(f"✓ Wczytano champion_roles.csv: {len(df_roles)} wierszy")
    if df_counters is not None:
        print(f"✓ Wczytano counters.csv: {len(df_counters)} wierszy")
    if df_matchups is not None:
        print(f"✓ Wczytano matchups_extra.csv: {len(df_matchups)} wierszy")
    
    print(f"\nGenerowanie wykresów do: {args.output_dir}")
    print("-" * 60)
    
    # Generuj wykresy
    plot_role_distribution(df_roles, args.output_dir)
    plot_counter_ratings_distribution(df_counters, args.output_dir)
    plot_strongest_counters(df_counters, args.output_dir, top_n=20)
    plot_most_countered_champions(df_counters, args.output_dir, top_n=15)
    plot_champion_counter_network(df_counters, args.output_dir, top_n=15)
    
    print("-" * 60)
    print(f"\n✅ Gotowe! Wszystkie wykresy zapisano w katalogu: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
