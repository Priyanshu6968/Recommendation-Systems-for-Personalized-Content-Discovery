import os
import yaml
import polars as pl
from pathlib import Path

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_netflix_file(file_path, output_path):
    print(f"Parsing {file_path}...")
    movie_ids = []
    customer_ids = []
    ratings = []
    dates = []
    
    current_movie_id = None
    count = 0
    
    with open(file_path, "r") as f:
        for line in f:
            count += 1
            if count % 5000000 == 0:
                print(f"Processed {count} lines...")
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(",")
                if len(parts) == 3:
                    movie_ids.append(current_movie_id)
                    customer_ids.append(int(parts[0]))
                    ratings.append(int(parts[1]))
                    dates.append(parts[2])
                    
    df = pl.DataFrame({
        "movie_id": movie_ids,
        "customer_id": customer_ids,
        "rating": pl.Series(ratings, dtype=pl.UInt8),
        "date": dates
    })
    
    # Cast date string to Date type
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    
    df.write_parquet(output_path)
    print(f"Saved to {output_path}")

def main():
    config = load_config()
    raw_path = Path(config["data"]["raw_path"])
    interim_path = Path(config["data"]["interim_path"])
    
    interim_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, 5):
        file_name = f"combined_data_{i}.txt"
        file_path = raw_path / file_name
        output_path = interim_path / f"data_{i}.parquet"
        
        if file_path.exists():
            parse_netflix_file(file_path, output_path)
        else:
            print(f"Warning: {file_path} not found.")

if __name__ == "__main__":
    main()
