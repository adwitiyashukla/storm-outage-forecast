from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .refdata import states


def ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    if len(ring) < 3:
        return 0.0, 0.0, 0.0
    lat0 = sum(point[1] for point in ring) / len(ring)
    scale = math.cos(math.radians(lat0))
    area = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        a1, a2 = x1 * scale, x2 * scale
        cross = a1 * y2 - a2 * y1
        area += cross
        cx += (a1 + a2) * cross
        cy += (y1 + y2) * cross
    if abs(area) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    return cx / (3 * area) / scale, cy / (3 * area), abs(area) / 2


def feature_centroid(geometry: dict) -> tuple[float, float]:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"unsupported geometry {geometry['type']}")
    total = sx = sy = 0.0
    for polygon in polygons:
        ring = [list(map(float, point[:2])) for point in polygon[0]]
        x, y, area = ring_centroid(ring)
        weight = area if area > 0 else 1e-9
        total += weight
        sx += x * weight
        sy += y * weight
    return sx / total, sy / total


def read_geojson(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("type") != "FeatureCollection" or not data.get("features"):
        raise ValueError(f"{path} is not a county FeatureCollection")
    return data


def county_frame(geo: dict) -> pd.DataFrame:
    conus = set(states()["state_fips"])
    rows = []
    for feature in geo["features"]:
        geo_id = str(feature["id"]).zfill(5)
        if geo_id[:2] not in conus:
            continue
        lon, lat = feature_centroid(feature["geometry"])
        props = feature.get("properties", {})
        rows.append(
            {
                "geo_id": geo_id,
                "state_fips": geo_id[:2],
                "name": str(props.get("NAME", geo_id)),
                "lat": lat,
                "lon": lon,
                "land_sqmi": float(props.get("CENSUSAREA") or 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("geo_id").reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def unit_vectors(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord_km(radius_km: float) -> float:
    return 2 * 6371.0 * math.sin(radius_km / (2 * 6371.0))
