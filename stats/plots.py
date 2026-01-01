"""
Wizualizacja danych z plików CSV wygenerowanych przez counter_stats.py
Tworzy różne wykresy do analizy champion statistics, roles, win rates, counters itp.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


def load_data(data_dir: str):
    """Wczytuje pliki CSV z katalogu"""
    roles_path = os.path.join(data_dir, "champion_roles.csv")
    counters_path = os.path.join(data_dir, "counters.csv")
    
    df_roles = pd.read_csv(roles_path) if os.path.exists(roles_path) else None
    df_counters = pd.read_csv(counters_path) if os.path.exists(counters_path) else None
    
    return df_roles, df_counters


def _has_data(df, name: str) -> bool:
    if df is None or df.empty:
        print(f"Brak danych {name}, pomijam")
        return False
    return True


def _save(output_dir: str, filename: str):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    print(f"✓ Zapisano: {filename}")
    plt.close()


def plot_winrate_vs_pickrate_scatter(df_roles, output_dir):
    if not _has_data(df_roles, "champion_roles"):
        return

    df_valid = df_roles[(df_roles['win_rate_pct'].notna()) & (df_roles['pick_rate_pct'].notna())]
    if df_valid.empty:
        print("Brak pełnych danych do scattera win/pick")
        return

    plt.figure(figsize=(12, 8))
    roles = df_valid['role'].unique()
    colors = sns.color_palette("Set2", len(roles))

    for role, color in zip(roles, colors):
        df_role = df_valid[df_valid['role'] == role]
        plt.scatter(df_role['pick_rate_pct'], df_role['win_rate_pct'],
                    label=role, alpha=0.65, s=90, color=color,
                    edgecolors='black', linewidth=0.5)

    plt.xlabel('Pick Rate (%)', fontsize=11)
    plt.ylabel('Win Rate (%)', fontsize=11)
    plt.title('Win rate vs pick rate (role)', fontsize=13, fontweight='bold')
    plt.legend(title='Rola', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    _save(output_dir, '1_winrate_vs_pickrate.png')


def plot_role_distribution(df_roles, output_dir):
    if not _has_data(df_roles, "champion_roles"):
        return

    role_counts = df_roles['role'].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(role_counts.index, role_counts.values, color=sns.color_palette("pastel"))
    plt.xlabel('Rola')
    plt.ylabel('Liczba championów')
    plt.title('Rozkład championów według ról', fontsize=13, fontweight='bold')
    plt.tick_params(axis='x', rotation=30)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 int(height), ha='center', va='bottom', fontweight='bold')

    _save(output_dir, '1_role_distribution.png')


def plot_strongest_counters(df_counters, output_dir, top_n=15):
    if not _has_data(df_counters, "counters"):
        return

    df_top = df_counters[df_counters['counter_rating'].notna()].nlargest(top_n, 'counter_rating')
    if df_top.empty:
        print("Brak wartości counter_rating")
        return

    labels = df_top.apply(lambda row: f"{row['champion']} ({row['role']}) vs {row['counter']}", axis=1)

    plt.figure(figsize=(12, 9))
    bars = plt.barh(labels, df_top['counter_rating'], color=sns.color_palette("flare", len(df_top)))
    plt.xlabel('Counter rating')
    plt.title(f'Top {top_n} najsilniejszych counterów', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.4, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}', va='center', fontsize=9)

    _save(output_dir, '2_strongest_counters.png')


def summarize_counters_table(df_counters, output_dir, top_n=3):
    """Eksportuje tabelę: dla każdego championa/roli najlepsze i najgorsze counter-picki."""
    if not _has_data(df_counters, "counters"):
        return

    df = df_counters.copy()

    def format_label(row):
        if pd.notna(row.counter_win_rate_pct):
            return f"{row.counter} ({row.counter_win_rate_pct:.1f}% win)"
        if pd.notna(row.counter_rating):
            return f"{row.counter} (rating {row.counter_rating:.1f})"
        return row.counter

    rows = []
    grouped = df.groupby(['champion', 'role'])

    for (champ, role), group in grouped:
        work = group.copy()
        work['sort_score'] = work['counter_win_rate_pct']
        mask = work['sort_score'].isna()
        work.loc[mask, 'sort_score'] = work.loc[mask, 'counter_rating']
        work = work[work['sort_score'].notna()]
        if work.empty:
            continue

        best = work.nsmallest(top_n, 'sort_score')  # najniższy win% => korzystne dla championa
        worst = work.nlargest(top_n, 'sort_score')   # najwyższy win% => trudne matchupy

        best_str = ", ".join(format_label(r) for r in best.itertuples()) or "-"
        worst_str = ", ".join(format_label(r) for r in worst.itertuples()) or "-"

        rows.append({
            'champion': champ,
            'role': role,
            'best_against': best_str,
            'worst_against': worst_str,
        })

    if not rows:
        print("Brak danych do tabeli counterów")
        return

    summary_df = pd.DataFrame(rows).sort_values(['champion', 'role'])

    csv_path = os.path.join(output_dir, 'counters_summary.csv')
    summary_df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, 'counters_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(summary_df.to_markdown(index=False))

    print(f"✓ Zapisano: counters_summary.csv ({len(summary_df)} wierszy)")
    print(f"✓ Zapisano: counters_summary.md")
    print(summary_df.head(5).to_string(index=False))


def plot_counter_winrate_box_by_role(df_counters, output_dir):
    """Boxplot win rate'ów counterów na championów w podziale na role."""
    if not _has_data(df_counters, "counters"):
        return

    df_valid = df_counters[df_counters['counter_win_rate_pct'].notna()]
    if df_valid.empty:
        print("Brak wartości counter_win_rate_pct do boxplota")
        return

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_valid, x='role', y='counter_win_rate_pct', palette="coolwarm", width=0.55)
    sns.stripplot(data=df_valid, x='role', y='counter_win_rate_pct', color='black', size=3, alpha=0.25)
    plt.axhline(50, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Rola')
    plt.ylabel('Win rate countera (%)')
    plt.title('Rozkład win rate counterów vs championów per rola', fontsize=13, fontweight='bold')
    plt.tick_params(axis='x', rotation=25)

    _save(output_dir, '3_counter_winrate_by_role.png')


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
    df_roles, df_counters = load_data(args.data_dir)
    
    if df_roles is not None:
        print(f"✓ Wczytano champion_roles.csv: {len(df_roles)} wierszy")
    if df_counters is not None:
        print(f"✓ Wczytano counters.csv: {len(df_counters)} wierszy")
    print(f"\nGenerowanie wykresów do: {args.output_dir}")
    print("-" * 60)
    
    # Generuj wykresy
    plot_winrate_vs_pickrate_scatter(df_roles, args.output_dir)
    plot_role_distribution(df_roles, args.output_dir)
    plot_strongest_counters(df_counters, args.output_dir, top_n=15)
    summarize_counters_table(df_counters, args.output_dir, top_n=3)
    plot_counter_winrate_box_by_role(df_counters, args.output_dir)
    
    print("-" * 60)
    print(f"\n✅ Gotowe! Wszystkie wykresy zapisano w katalogu: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
