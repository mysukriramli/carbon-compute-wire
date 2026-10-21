#!/usr/bin/env python3
"""
Autonomous Carbon-Compute Routing Wire
Cross-references live GPU spot pricing with local meteorological data 
(solar/wind) to calculate optimal Eco-Compute workload routing for AI agents.
"""

import os
import json
import logging
import urllib.parse
from datetime import datetime, timezone
import requests
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)


class EcoRoutingNode(BaseModel):
    machine_id: int
    gpu_model: str
    num_gpus: int
    price_usd_per_hour: float
    location_country: str
    latitude: float
    longitude: float
    wind_speed_kmh: float
    solar_irradiance_wm2: float
    eco_compute_score: float
    agent_directive: str


def fetch_cheapest_gpus() -> list:
    """Fetches the cheapest available high-end GPUs from Vast.ai."""
    # Query for verified, unallocated H100s, A100s, and RTX 4090s
    query = '{"gpu_name":{"in":["RTX 4090", "A100", "A100 SXM4", "H100"]},"verified":{"eq":true},"allocated":{"eq":false}}'
    encoded = urllib.parse.quote(query)
    url = f"https://console.vast.ai/api/v0/bundles/?q={encoded}"
    
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        offers = res.json().get("offers", [])
        
        # Deduplicate to find the single cheapest instance per country/GPU pair
        hubs = {}
        for o in offers:
            gpu = o.get("gpu_name")
            country = o.get("geolocation", "Unknown")
            price = o.get("dph_total")
            
            if not gpu or not price or not country:
                continue
                
            key = f"{gpu}_{country}"
            if key not in hubs or price < hubs[key]["price"]:
                hubs[key] = {
                    "machine_id": o.get("machine_id"),
                    "gpu": gpu,
                    "num_gpus": o.get("num_gpus"),
                    "price": price,
                    "country": country,
                    "lat": o.get("client_lat", 0.0), # Approximate data center latitude
                    "lon": o.get("client_lon", 0.0)
                }
        return list(hubs.values())
    except Exception as e:
        logging.error(f"Failed to fetch GPU market data: {e}")
        return []


def fetch_weather_for_location(lat: float, lon: float) -> dict:
    """Fetches real-time solar and wind data for the data center coordinates."""
    # Open-Meteo provides free, no-key global weather metrics
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=wind_speed_10m,direct_radiation"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        current = res.json().get("current", {})
        return {
            "wind_kmh": current.get("wind_speed_10m", 0.0),
            "solar_wm2": current.get("direct_radiation", 0.0) # Watts per square meter
        }
    except Exception as e:
        logging.warning(f"Failed to fetch weather for {lat},{lon}: {e}")
        return {"wind_kmh": 0.0, "solar_wm2": 0.0}


def synthesize_directive(gpu: str, country: str, price: float, score: float) -> str:
    """Generates the routing rule for downstream MLOps agents."""
    if score > 80:
        return f"PRIORITY ROUTING: Deploy {gpu} workloads to {country}. Exceptional renewable generation currently offsetting the ${price:.2f}/hr spot price."
    elif score > 40:
        return f"STANDARD ROUTING: Viable {gpu} node in {country} at ${price:.2f}/hr. Average grid renewable mix."
    else:
        return f"RESTRICT ROUTING: Avoid {country} unless urgent. Low renewable generation spikes carbon footprint despite ${price:.2f}/hr rate."


def run_pipeline():
    logging.info("Starting Eco-Compute Cross-Reference Pipeline...")
    gpu_hubs = fetch_cheapest_gpus()
    
    records = []
    md_lines = [
        "# Autonomous Green-Compute Routing Radar",
        "> Merges real-time GPU spot pricing with live solar/wind meteorological data to optimize MLOps carbon footprints.\n",
        "## Active Workload Routing Directives"
    ]

    for hub in gpu_hubs:
        # Prevent API spam by skipping invalid coordinates
        if hub["lat"] == 0.0 and hub["lon"] == 0.0:
            continue
            
        weather = fetch_weather_for_location(hub["lat"], hub["lon"])
        
        # Calculate Eco-Compute Score (0 to 100)
        # High solar (>500 W/m2) or high wind (>20 km/h) combined with low GPU price increases the score
        renewable_factor = min((weather["solar_wm2"] / 800) + (weather["wind_kmh"] / 30), 1.5)
        price_inversion = 1.0 / max(hub["price"], 0.1)
        raw_score = (renewable_factor * price_inversion) * 50
        eco_score = min(max(round(raw_score, 1), 0.0), 100.0)

        directive = synthesize_directive(hub["gpu"], hub["country"], hub["price"], eco_score)
        
        record = EcoRoutingNode(
            machine_id=hub["machine_id"],
            gpu_model=hub["gpu"],
            num_gpus=hub["num_gpus"],
            price_usd_per_hour=round(hub["price"], 3),
            location_country=hub["country"],
            latitude=hub["lat"],
            longitude=hub["lon"],
            wind_speed_kmh=weather["wind_kmh"],
            solar_irradiance_wm2=weather["solar_wm2"],
            eco_compute_score=eco_score,
            agent_directive=directive
        )
        records.append(record)

    # Sort so the highest Eco-Score appears first
    records.sort(key=lambda x: x.eco_compute_score, reverse=True)

    for r in records[:20]:
        md_lines.extend([
            f"### {r.gpu_model} Cluster in {r.location_country} (Eco-Score: **{r.eco_compute_score}**/100)",
            f"- **Spot Rate:** ${r.price_usd_per_hour}/hr | **Node ID:** `{r.machine_id}`",
            f"- **Local Grid Weather:** Wind {r.wind_speed_kmh} km/h | Solar {r.solar_irradiance_wm2} W/m²",
            f"- **Agent Directive:** {r.agent_directive}\n"
        ])

    if not records:
        logging.warning("No viable GPU/Weather pairs found today.")
        return

    now = datetime.now(timezone.utc)
    payload = {
        "metadata": {
            "generated_at": now.isoformat(),
            "data_sources": ["Vast.ai GPU Market", "Open-Meteo Global Weather"],
            "schema_version": "1.0.0"
        },
        "routing_nodes": [r.model_dump() for r in records]
    }

    # Write rolling endpoints
    with open(os.path.join(DATA_DIR, "today.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(DATA_DIR, "llms.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Write permanent archive
    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y"), now.strftime("%m"))
    os.makedirs(month_dir, exist_ok=True)
    with open(os.path.join(month_dir, f"{now.strftime('%Y-%m-%d')}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info("Carbon-Compute routing endpoints successfully published.")

if __name__ == "__main__":
    run_pipeline()
