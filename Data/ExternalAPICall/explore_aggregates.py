# Data/ExternalAPICall/explore_aggregates.py
import datetime
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from Data.ExternalAPICall.euskalmetAPI import generate_jwt

BASE = "https://api.euskadi.eus/euskalmet"
CATALOG = BASE_DIR / "Data" / "GeoJSON" / "euskalmet_stations.json"

day = datetime.date.today() - datetime.timedelta(days=1)
headers = {"Accept": "application/json", "Authorization": f"Bearer {generate_jwt()}"}

stations = [
    s for s in json.loads(CATALOG.read_text(encoding="utf-8"))
    if s["type"] == "METEOROLOGICAL"
]
print(f"{len(stations)} estaciones meteorológicas, fecha {day}")

ok, fail = [], []
stations_with_measure = defaultdict(set)
value_stats = {}
sample_saved = False

for s in stations:
    sid = s["stationId"]
    url = (f"{BASE}/readings/aggregated/byDay/forStation/{sid}"
           f"/at/{day.year}/{day.month:02d}/{day.day:02d}")
    try:
        r = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        fail.append((sid, f"error de red: {e}"))
        continue
    if r.status_code != 200:
        fail.append((sid, r.status_code))
        continue

    r.encoding = "utf-8"
    data = r.json()
    ok.append(sid)

    if not sample_saved:
        (BASE_DIR / "Data" / "GeoJSON" / f"sample_aggregate_{sid}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sample_saved = True

    for item in data.get("items", []):
        key = (item["measureType"], item["measureId"])
        stations_with_measure[key].add(sid)
        vals = [v for rd in item.get("readings", []) for v in rd.get("values", []) if v is not None]
        if vals and key not in value_stats:
            value_stats[key] = (sid, item["sensorId"], len(vals), min(vals), max(vals))
    time.sleep(0.2)

print(f"\nCon datos: {len(ok)} | Sin datos: {len(fail)}")
print("Fallos:", fail[:15])

print("\nmeasureType / measureId  [estaciones]  (ejemplo: estación, sensor, nº valores, min, max)")
for key, sids in sorted(stations_with_measure.items(), key=lambda kv: -len(kv[1])):
    print(f"{key[0]:24s} {key[1]:28s} {len(sids):3d}   {value_stats.get(key)}")