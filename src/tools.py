"""
tools.py - Geospatial tools for use by an LLM or scripts.

Functions:
- search_stac
- compute_ndvi_for_item

Dependencies:
  pystac-client
  planetary-computer
  rioxarray
  geopandas
  shapely
  xarray, numpy
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import box, mapping

from pystac_client import Client
import planetary_computer as pc

# ---------------------------------------------------------------------
# STAC setup
# ---------------------------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
catalog = Client.open(STAC_URL)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _compute_ndvi(nir_arr, red_arr):
    nir_f = nir_arr.astype("float32")
    red_f = red_arr.astype("float32")
    ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-6)
    return ndvi


# ---------------------------------------------------------------------
# Tool 1: Search STAC
# ---------------------------------------------------------------------

def search_stac(
    bbox: List[float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_items: int = 5,
    cloud_lt: float = 30
) -> List[Dict]:
    """
    Search Sentinel-2 scenes over bbox.

    Args:
        bbox: [minx, miny, maxx, maxy] in WGS84
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        max_items: # max results
        cloud_lt: cloud cover threshold

    Returns:
        list of dicts with basic metadata
    """

    start = start_date or (datetime.utcnow().date() - timedelta(days=90)).isoformat()
    end = end_date or datetime.utcnow().date().isoformat()

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_lt}},
        limit=max_items,
    )

    items = list(search.get_items())

    results = []
    for it in items:
        results.append({
            "id": it.id,
            "datetime": it.properties.get("datetime"),
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "assets": list(it.assets.keys()),
        })

    return results


# ---------------------------------------------------------------------
# Tool 2: Compute NDVI for an item (optionally clip)
# ---------------------------------------------------------------------

def compute_ndvi_for_item(
    item_id: str,
    clip_bbox: Optional[List[float]] = None
) -> Dict:
    """
    Compute NDVI stats for a single STAC item.

    Args:
        item_id: STAC id
        clip_bbox: optional bounding box [minx, miny, maxx, maxy], WGS84

    Returns:
        dict {mean, min, max}
    """

    # get item
    it = catalog.get_item(item_id)

    if not it:
        raise ValueError(f"Item {item_id} not found.")

    # get red & nir
    red_href = pc.sign(it.assets["B04"].href)
    nir_href = pc.sign(it.assets["B08"].href)

    red = rioxarray.open_rasterio(red_href).squeeze()
    nir = rioxarray.open_rasterio(nir_href).squeeze()

    ndvi = _compute_ndvi(nir, red)
    ndvi.name = "NDVI"

    # clip if requested
    if clip_bbox:
        gdf = gpd.GeoDataFrame(
            {"geometry": [box(*clip_bbox)]},
            crs="EPSG:4326"
        ).to_crs(ndvi.rio.crs)

        ndvi = ndvi.rio.clip(
            gdf.geometry.apply(mapping),
            gdf.crs,
            drop=True,
            invert=False
        )

    return {
        "mean": float(ndvi.mean().values),
        "min": float(ndvi.min().values),
        "max": float(ndvi.max().values)
    }
