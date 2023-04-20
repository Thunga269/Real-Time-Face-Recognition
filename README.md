# FaceID
## Update code

```bash
git checkout dev
git pull origin dev
```

## Download models

- Chạy file sau để tải models:

```bash
chmod 777 scripts/download_weights.sh
sh scripts/download_weights.sh
```

## Cài đặt

```bash
# Tao mot moi truong ao conda
conda create --name face_id python=3.7.6
conda activate face_id

# Cai dat thu vien
pip install -r requirements.txt

## API
python3 main.py
```