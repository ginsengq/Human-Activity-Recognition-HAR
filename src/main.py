import os
import zipfile
import pandas as pd
from pathlib import Path
import pyarrow
import fastparquet

os.environ["KAGGLE_USERNAME"] = "username"
os.environ["KAGGLE_KEY"] = "KGAT_98e9ca998565fbaf404ed15955cb9977"

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


api.dataset_download_files(
    "romanchereshnev/hugadb-human-gait-database",
    path="data/hugadb-human-gait-database",
    unzip=False
)

zip_path = Path(r"C:\Users\asdf\Desktop\hugadb_project\data\hugadb-human-gait-database\hugadb-human-gait-database.zip")

all_dataframes = []

print("Открываю архив...")

with zipfile.ZipFile(zip_path, 'r') as archive:
    all_files = archive.namelist()
    
    csv_files = [f for f in all_files if f.endswith('.csv') and 'HuGaDB' in f]
    
    print(f"Найдено файлов для склейки: {len(csv_files)}")
    
    for file_name in csv_files:
        clean_name = Path(file_name).name 
        
       
        name_parts = clean_name.split('_')
        
        try:
            subject_id = int(name_parts[3])
            session_id = int(name_parts[4].replace('.csv', ''))
        except (IndexError, ValueError):
            continue 
            
        with archive.open(file_name) as file:
            df = pd.read_csv(file)
            
            df['subject_id'] = subject_id
            df['session_id'] = session_id
            
            all_dataframes.append(df)

print("Таблицы в единый DataFrame...")

full_df = pd.concat(all_dataframes, ignore_index=True)

float_cols = full_df.select_dtypes(include=['float64']).columns
full_df[float_cols] = full_df[float_cols].astype('float32')

print(f"Итоговый размер данных: {full_df.shape}")

full_df.to_parquet("hugadb_cleaned.parquet", index=False)
