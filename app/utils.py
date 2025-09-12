import json, os, tempfile, shutil
from datetime import datetime, timezone, timedelta

JAKARTA_TZ = timezone(timedelta(hours=7))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json_atomic(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp_name = tf.name
    shutil.move(tmp_name, path)

def now_jkt_iso() -> str:
    return datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")
