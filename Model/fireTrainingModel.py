"""
Generate Non-Fire Weather Observations from Euskalmet (Incremental Saver).

Samples random days evenly across all seasons, queries Euskalmet API,
assigns fire_risk = 0, and incrementally appends to real_weather_training.csv.
"""

import argparse
import datetime
from pathlib import Path
import random
import sys
import time

import pandas as pd

# Path setup: BASE_DIR points to project root (FireModelProject)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import Euskalmet API helper from Data/ExternalAPICall/
from Data.ExternalAPICall.euskalmetAPI import get_variables_for_specific_day

CSV_PATH = BASE_DIR / "Data" / "MITECO" / "real_weather_training.csv"


def generate_random_dates(start_date: datetime.date, end_date: datetime.date, num_days: int) -> list[datetime.date]:
    """Generates unique random dates within a date range."""
    total_days = (end_date - start_date).days
    if total_days <= 0:
        raise ValueError("start_date must be before end_date")
    
    sampled_offsets = random.sample(range(total_days), min(num_days, total_days))
    sampled_dates = [start_date + datetime.timedelta(days=offset) for offset in sampled_offsets]
    return sorted(sampled_dates)


def update_training_csv(new_df: pd.DataFrame, csv_path: Path = CSV_PATH) -> None:
    """Merges new non-fire samples into existing training CSV and removes duplicates."""
    if new_df.empty:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Drop duplicate records based on date and municipality
    combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.strftime("%Y-%m-%d")
    combined_df = combined_df.drop_duplicates(subset=["date", "municipality"], keep="last")

    # Show updated dataset stats
    fire_count = (combined_df["fire_risk"] == 1).sum()
    non_fire_count = (combined_df["fire_risk"] == 0).sum()
    total = len(combined_df)
    pos_rate = fire_count / total if total > 0 else 0.0

    combined_df.to_csv(csv_path, index=False)
    print(f"   💾 Saved to CSV | Total: {total} | Fire: {fire_count} | Non-Fire: {non_fire_count} | Pos Rate: {pos_rate:.1%}")


def fetch_non_fire_dataset(num_days: int = 60, start_str: str = "2024-01-01") -> None:
    """Fetches non-fire weather records for randomly selected dates and saves after each day."""
    start_date = datetime.date.fromisoformat(start_str)
    end_date = datetime.date.today() - datetime.timedelta(days=2)
    
    selected_dates = generate_random_dates(start_date, end_date, num_days)
    print(f"Sampling {len(selected_dates)} random dates between {start_date} and {end_date}...\n")
    
    for idx, day in enumerate(selected_dates, start=1):
        print(f"[{idx}/{len(selected_dates)}] Fetching Euskalmet weather for {day.isoformat()}...")
        try:
            df_day = get_variables_for_specific_day(day)
            if df_day is not None and not df_day.empty:
                df_day["fire_risk"] = 0  # Label as non-fire day
                update_training_csv(df_day)
            else:
                print(f"   ⚠️ No data returned for {day.isoformat()}, skipping.")
        except Exception as err:
            print(f"   ❌ Error fetching {day.isoformat()}: {err}")
        
        time.sleep(0.2)

    print("\n🎉 Sampling finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch non-fire weather samples from Euskalmet.")
    parser.add_argument("--days", type=int, default=60, help="Number of random dates to sample (default: 60)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Earliest date to sample YYYY-MM-DD (default: 2024-01-01)")
    args = parser.parse_args()

    fetch_non_fire_dataset(num_days=args.days, start_str=args.start)