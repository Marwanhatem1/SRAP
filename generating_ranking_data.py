import numpy as np
import pandas as pd

np.random.seed(42)

# Haversine formula to compute real GPS distance in kilometers
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return np.round(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), 2)

# 1. GENERATE MASTER PROVIDERS TABLE (Greater Cairo Bounding Box)
def generate_providers(num_providers=250):
    providers = []
    for pid in range(1, num_providers + 1):
        # 0: Mechanic (45%), 1: Tow Truck (30%), 2: Spare Parts Store (25%)
        service_type = np.random.choice([0, 1, 2], p=[0.45, 0.30, 0.25])
        
        # Cairo & Giza GPS coordinates (Approx: Lat 29.95 to 30.15, Lon 31.15 to 31.45)
        lat = np.round(np.random.uniform(29.9500, 30.1500), 5)
        lon = np.round(np.random.uniform(31.1500, 31.4500), 5)
        
        # Capacity tier: 1 (Light), 2 (Medium), 3 (Heavy)
        capacity_tier = np.random.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
        rating = np.round(np.clip(np.random.normal(4.15, 0.55), 1.5, 5.0), 1)
        
        # Mobile workshop applies primarily to mechanics (1 = Full Workshop, 0 = Traditional Tools)
        mobile_workshop = np.random.choice([0, 1], p=[0.40, 0.60]) if service_type == 0 else 0
        
        providers.append({
            "provider_id": pid,
            "service_type": service_type,
            "lat": lat,
            "lon": lon,
            "capacity_tier": capacity_tier,
            "rating": rating,
            "mobile_workshop": mobile_workshop,
            "is_available": np.random.choice([True, False], p=[0.85, 0.15])
        })
    return pd.DataFrame(providers)

# 2. SIMULATE BREAKDOWN REQUESTS & CANDIDATE RANKINGS
def generate_ranking_requests(providers_df, num_requests=1000):
    rows = []
    # Fair baseline prices (EGP) by (service_type, user_vehicle_tier)
    base_prices = {
        0: {1: 300, 2: 450, 3: 900},   # Mechanic base repair cost
        1: {1: 450, 2: 700, 3: 1500},  # Tow Truck base cost
        2: {1: 600, 2: 950, 3: 2200}   # Spare Part average cost
    }

    available_providers = providers_df[providers_df["is_available"] == True]

    for req_id in range(1, num_requests + 1):
        req_service = np.random.choice([0, 1, 2], p=[0.45, 0.30, 0.25])
        user_tier = np.random.choice([1, 2, 3], p=[0.60, 0.25, 0.15])
        user_lat = np.random.uniform(29.9600, 30.1400)
        user_lon = np.random.uniform(31.1600, 31.4400)

        # STAGE 1 HARD FILTER: Same service type AND capacity_tier >= user_tier
        valid_pool = available_providers[
            (available_providers["service_type"] == req_service) &
            (available_providers["capacity_tier"] >= user_tier)
        ]

        # Sample 6 to 12 competing candidates for this request
        num_cands = min(len(valid_pool), np.random.randint(6, 13))
        candidates = valid_pool.sample(n=num_cands, replace=False).copy()

        fair_market_price = base_prices[req_service][user_tier] * np.random.uniform(0.90, 1.10)

        for _, prov in candidates.iterrows():
            dist_km = haversine_km(user_lat, user_lon, prov["lat"], prov["lon"])
            tier_diff = int(prov["capacity_tier"] - user_tier)

            # Simulate price ratio (some providers charge fair rates, some exploit)
            price_ratio = np.round(np.random.lognormal(mean=0.02 + 0.12 * tier_diff, sigma=0.20), 2)
            quoted_price = np.round(fair_market_price * price_ratio, 0)

            # Calculate human preference utility score
            # 1. Distance importance is highest for Tow Trucks & Mechanics, slightly lower for Spare Parts
            dist_weight = 42.0 if req_service in [0, 1] else 28.0
            utility = dist_weight * np.exp(-0.14 * dist_km)

            # 2. Rating bonus
            utility += 25.0 * ((prov["rating"] / 5.0) ** 2)

            # 3. Anti-exploitation price fairness rule
            if price_ratio <= 1.15:
                utility += 25.0 * (1.30 - price_ratio)
            elif price_ratio <= 1.30:
                utility += 5.0
            else:
                utility -= 35.0  # Strong penalty for exploitative pricing

            # 4. Penalize oversized tow trucks when an exact tier match exists
            if req_service == 1:
                utility -= 10.0 * tier_diff

            # 5. Bonus for Mechanics with a full Mobile Workshop
            if req_service == 0 and prov["mobile_workshop"] == 1:
                utility += 12.0

            # Add realistic noise
            utility += np.random.normal(0, 2.5)

            rows.append({
                "request_id": req_id,
                "provider_id": int(prov["provider_id"]),
                "service_type": int(req_service),
                "user_vehicle_tier": int(user_tier),
                "tier_difference": tier_diff,
                "distance_km": dist_km,
                "rating": prov["rating"],
                "quoted_price": quoted_price,
                "price_ratio": price_ratio,
                "mobile_workshop": int(prov["mobile_workshop"]),
                "_utility": utility
            })

    df = pd.DataFrame(rows)

    # Convert continuous utility into ordinal relevance grades (0, 1, 2, 3) per request group
    ranks = df.groupby("request_id")["_utility"].rank(ascending=False, method="first")
    df["relevance_label"] = np.select(
        [
            (df["price_ratio"] > 1.30) | (df["_utility"] < 25.0),  # Exploitative or unsuitable option
            ranks == 1,                                             # Top recommendation
            ranks <= 3,                                             # Strong alternative
        ],
        [0, 3, 2],
        default=1,                                                  # Acceptable backup
    )

    return df.drop(columns=["_utility"]).sort_values("request_id").reset_index(drop=True)

if __name__ == "__main__":
    providers_df = generate_providers(250)
    providers_df.to_csv("srap_providers_db.csv", index=False)

    ranking_df = generate_ranking_requests(providers_df, num_requests=1000)

    # Split strictly by request_id (80% Train, 20% Test)
    train_df = ranking_df[ranking_df["request_id"] <= 800]
    test_df = ranking_df[ranking_df["request_id"] > 800]

    train_df.to_csv("srap_ranking_train.csv", index=False)
    test_df.to_csv("srap_ranking_test.csv", index=False)

    print(f"Saved srap_providers_db.csv ({len(providers_df)} providers)")
    print(f"Saved srap_ranking_train.csv ({len(train_df)} rows across 800 requests)")
    print(f"Saved srap_ranking_test.csv  ({len(test_df)} rows across 200 requests)")
    print("\nRelevance Label Distribution in Training Set:")
    print(train_df["relevance_label"].value_counts().sort_index())
