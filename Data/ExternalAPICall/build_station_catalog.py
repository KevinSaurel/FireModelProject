

    # Data/ExternalAPICall/build_station_catalog.py
import json
import time
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from Data.ExternalAPICall.euskalmetAPI import generate_jwt

BASE = "https://api.euskadi.eus/euskalmet"
headers = {"Accept": "application/json", "Authorization": f"Bearer {generate_jwt()}"}

ids = sorted({s["stationId"] for s in requests.get(f"{BASE}/stations", headers=headers, timeout=20).json()})

catalog = []
for i, sid in enumerate(ids, 1):
    r = requests.get(f"{BASE}/stations/{sid}/current", headers=headers, timeout=20)
    if r.status_code != 200:
        print(f"{sid}: HTTP {r.status_code}")
        continue
    d = r.json()
    catalog.append({
        "stationId": sid,
        "type": d.get("stationType"),
        "name": d.get("name", {}).get("SPANISH"),
        "municipality": d.get("municipality", {}).get("SPANISH"),
        "province": d.get("province", {}).get("SPANISH"),
        "lat": d["position"]["position"]["GOOGLE"]["y"],
        "lon": d["position"]["position"]["GOOGLE"]["x"],
        "sensors": [s["sensorKey"].split("/")[-1] for s in d.get("sensors", [])],
    })
    time.sleep(0.2)

out = BASE_DIR / "Data" / "euskalmet_stations.json"
out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

from collections import Counter
print(Counter(s["type"] for s in catalog))
print(Counter(k[:2] for s in catalog for k in s["sensors"]))