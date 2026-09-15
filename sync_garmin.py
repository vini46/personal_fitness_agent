import os
import json
import base64
import io
import tarfile
from datetime import datetime, timedelta, timezone
from garminconnect import Garmin
import garth

def restore_session():
    # Decodes base64 secret back into ~/.garth
    b64_tokens = os.environ.get("GARMIN_TOKENS_BASE64")
    if not b64_tokens:
        raise ValueError("GARMIN_TOKENS_BASE64 secret is missing!")
    
    compressed_data = base64.b64decode(b64_tokens)
    buf = io.BytesIO(compressed_data)
    
    token_dir = os.path.expanduser("~")
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=token_dir)
    
    garth.resume("~/.garth")
    client = Garmin()
    client.login()
    return client

def filter_hr_spikes(hr_list, threshold=15):
    """Filters single-sample optical HR spikes."""
    if not hr_list:
        return []
    
    cleaned = list(hr_list)
    threw_out = 0
    
    for i in range(1, len(cleaned) - 1):
        prev_val = cleaned[i-1]
        curr_val = cleaned[i]
        next_val = cleaned[i+1]
        
        if prev_val is not None and curr_val is not None and next_val is not None:
            if (curr_val - prev_val > threshold) and (curr_val - next_val > threshold):
                cleaned[i] = round((prev_val + next_val) / 2)
                threw_out += 1

    print(f"Filtered out {threw_out} single-sample optical HR spikes.")
    return cleaned

def fetch_data():
    client = restore_session()
    
    # 1. Pull activities from last 180 days
    start_date = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
    activities = client.get_activities_by_date(start_date, datetime.now(timezone.utc).strftime("%Y-%m-%d"), "running")
    
    if not activities:
        print("No running activities found in window.")
        return None

    # Pick the latest quality/hard run or recent running activity
    latest_run = activities[0]
    activity_id = latest_run["activityId"]
    
    # Fetch details and splits
    details = client.get_activity_details(activity_id)
    splits = client.get_activity_splits(activity_id)
    
    # Map raw metrics
    descriptors = details.get("metricDescriptors", [])
    raw_metrics = details.get("activityDetailMetrics", [])
    
    metric_map = {d["key"]: d["metricsIndex"] for d in descriptors}
    
    hr_idx = metric_map.get("directHeartRate")
    speed_idx = metric_map.get("directSpeed")
    
    hr_stream = []
    speed_stream = []
    
    for row in raw_metrics:
        vals = row.get("metrics", [])
        hr_val = vals[hr_idx] if hr_idx is not None and hr_idx < len(vals) else None
        speed_val = vals[speed_idx] if speed_idx is not None and speed_idx < len(vals) else None
        
        hr_stream.append(hr_val)
        speed_stream.append(speed_val)

    clean_hr = filter_hr_spikes([h for h in hr_stream if h is not None])
    
    # Convert speeds (m/s) to pace (min/mile or min/km)
    # Min/mile = 26.8224 / speed_m_s
    paces_sec = []
    for s in speed_stream:
        if s and s > 0:
            paces_sec.append(26.8224 / s)

    processed_data = {
        "summary": {
            "activityId": activity_id,
            "name": latest_run.get("activityName"),
            "date": latest_run.get("startTimeLocal"),
            "distance_m": latest_run.get("distance"),
            "duration_s": latest_run.get("duration"),
            "avg_hr": latest_run.get("averageHR"),
            "max_hr": latest_run.get("maxHR"),
            "avg_speed_m_s": latest_run.get("averageSpeed"),
        },
        "splits": splits.get("lapDTOs", []),
        "per_second_clean_hr": clean_hr,
        "per_second_paces_min_mile": paces_sec,
    }
    
    with open("latest_run.json", "w") as f:
        json.dump(processed_data, f, indent=2)
        
    print("Saved activity data to latest_run.json")

if __name__ == "__main__":
    fetch_data()