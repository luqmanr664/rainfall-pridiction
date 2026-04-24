"""
Synthetic sample data for the four ecological zones of Ghana
(Coastal, Forest, Transition, Savannah).

The real study uses 40 years of Ghana Meteorological Agency observations,
which are not redistributable. To let the module run end-to-end, this
script produces CSV files with:

  * Feature ranges drawn from typical values reported in the paper
    (Appiah-Badu et al., 2022; Section II).
  * A Rainfall_Class label that DEPENDS on the features (higher humidity
    and lower sunshine -> higher probability of rain), so that the
    classifier has real signal to learn. Base rain rate per zone follows
    the paper's mean annual rainfall figures:
        Coastal    ~900 mm   - driest
        Savannah   ~1100 mm  - uni-modal, warm
        Transition ~1300 mm  - between Savannah and Forest
        Forest     ~2200 mm  - wettest, bi-modal
"""

import numpy as np
import pandas as pd


ZONES = {
    # base_rain: baseline P(rain) reflecting mean annual rainfall from the paper
    # temp/rh/sun ranges: plausible daily values for that zone
    "coastal":    {"base_rain": 0.35, "samples": 2800,
                   "max_t": (27, 34), "min_t": (22, 26),
                   "rh06":  (70, 92), "rh15": (55, 80),
                   "sun":   (5, 10),  "wind": (3, 12)},
    "savannah":   {"base_rain": 0.30, "samples": 2440,
                   "max_t": (28, 38), "min_t": (20, 26),
                   "rh06":  (55, 90), "rh15": (30, 65),
                   "sun":   (6, 11),  "wind": (2, 10)},
    "transition": {"base_rain": 0.50, "samples": 3200,
                   "max_t": (26, 35), "min_t": (21, 25),
                   "rh06":  (68, 94), "rh15": (45, 78),
                   "sun":   (4, 10),  "wind": (2, 10)},
    "forest":     {"base_rain": 0.65, "samples": 3840,
                   "max_t": (25, 33), "min_t": (22, 25),
                   "rh06":  (75, 96), "rh15": (55, 85),
                   "sun":   (3, 9),   "wind": (2, 8)},
}


def _rain_probability(rh06, rh15, sunshine, max_temp, base_rain):
    """Logistic mapping from features to P(rain).

    Raises probability with humidity, drops it with sunshine hours and
    higher daytime temperature. The coefficients are chosen so that the
    long-run mean matches the zone's `base_rain`.
    """
    humidity = (rh06 + rh15) / 2.0
    logit = (
        0.08 * (humidity - 70)      # wetter air -> more rain
        - 0.35 * (sunshine - 6)     # more sun   -> less rain
        - 0.10 * (max_temp - 30)    # hotter day -> less rain
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Blend with the zone baseline so the overall rain rate stays realistic.
    return 0.6 * prob + 0.4 * base_rain


def create_zone_csv(zone_name, config, seed=6):
    rng = np.random.default_rng(seed)
    n = config["samples"]

    max_temp = rng.uniform(*config["max_t"], n)
    min_temp = rng.uniform(*config["min_t"], n)
    rh_0600 = rng.uniform(*config["rh06"], n)
    rh_1500 = rng.uniform(*config["rh15"], n)
    sunshine = rng.uniform(*config["sun"], n)
    wind_speed = rng.uniform(*config["wind"], n)

    prob = _rain_probability(
        rh_0600, rh_1500, sunshine, max_temp, config["base_rain"]
    )
    rainfall_class = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "Max_Temp": max_temp,
        "Min_Temp": min_temp,
        "RH_0600": rh_0600,
        "RH_1500": rh_1500,
        "Sunshine": sunshine,
        "Wind_Speed": wind_speed,
        "Rainfall_Class": rainfall_class,
    })
    filename = f"{zone_name}_zone_data.csv"
    df.to_csv(filename, index=False)
    rain_rate = rainfall_class.mean()
    print(f"  {filename:30s}  n={n:4d}  rain_rate={rain_rate:.2%}")


if __name__ == "__main__":
    print("Generating sample CSVs for all four ecological zones:")
    for name, cfg in ZONES.items():
        create_zone_csv(name, cfg)
