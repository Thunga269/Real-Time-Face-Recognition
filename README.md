# FaceID
## Update code

```bash
git checkout dev
git pull origin dev
```

## Download models

Download models:

```bash
chmod 777 scripts/download_weights.sh
sh scripts/download_weights.sh
```

## Install

```bash
# Create env conda
conda create --name face_id python=3.7.6
conda activate face_id

  pip install -r requirements.txt

## API
python3 main.py
```
