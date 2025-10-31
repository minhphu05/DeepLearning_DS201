import os
# import plotly 
from dotenv import load_dotenv
# import plotly 
import torch
def download_from_kaggle ():
    """Downloads dataset from Kaggle"""
    # 1. Đọc file .env
    load_dotenv(dotenv_path="./.env")

    # 2. Tạo thư mục lưu dữ liệu (nếu chưa có)
    os.makedirs("./Lab_01/data/raw_data", exist_ok=True)

    # 3. Tải bộ dữ liệu từ Kaggle
    dataset = "hojjatk/mnist-dataset"
    output_dir = "./Lab_01/data/raw_data"

    cmd = f'kaggle datasets download -d {dataset} -p {output_dir} --unzip'
    os.system(cmd)

    print(f"Đã tải dữ liệu về thư mục: {output_dir}")
    
download_from_kaggle ()