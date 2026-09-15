import base64
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from garminconnect import Garmin
import garth


def restore_session() -> Garmin:
    """Restores Garmin session using garth directly or falls back to credentials."""
    b64_tokens = os.environ.get("GARMIN_TOKENS_BASE64")
    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")

    token_dir = None

    if b64_tokens:
        try:
            compressed_data = base64.b64decode(b64_tokens)
            buf = io.BytesIO(compressed_data)

            temp_extract = tempfile.mkdtemp()
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(path=temp_extract)

            token_dir = temp_extract
            if os.path.exists(os.path.join(temp_extract, ".garminconnect")):
                token_dir = os.path.join(temp_extract, ".garminconnect")

            print(f"Loading session tokens from: {token_dir}")
        except Exception as e:
            print(f"Failed to unpack GARMIN_TOKENS_BASE64: {e}")
            token_dir = None

    # 1. Try restoring via garth resume if token directory exists
    if token_dir and os.path.exists(token_dir):
        try:
            garth.resume(token_dir)
            api = Garmin()
            api.garth = garth.client
            print("Successfully authenticated using stored tokens!")
            return api
        except Exception as e:
            print(f"Session token resume failed: {e}. Falling back to credentials...")

    # 2. Fallback to Email + Password login
    if email and password:
        print("Authenticating using email and password...")
        api = Garmin(email, password)
        api.login()
        print("Successfully authenticated with credentials!")
        return api

    raise RuntimeError(
        "Failed to authenticate with Garmin: Tokens invalid/expired and no credentials provided."
    )

def filter_hr_spikes(hr_stream: list, threshold: int = 15) -> tuple[list, int]:
    """Filters single-sample optical wrist heart rate artifacts."""
    if not hr_stream:
        return [], 0

    cleaned = list(hr_stream)
    threw_out = 0

    for i in range(1, len(cleaned) - 1):
        prev_val = cleaned[i - 1]
        curr_val = cleaned[i]
        next_val = cleaned[i + 1]

        if prev_val is not None and curr_val is not None and next_val is not None:
            if (curr_val - prev_val > threshold) and (
                curr_val - next_val > threshold
            ):
                cleaned[i] = round((prev_val + next_val) / 2)
                threw_out += 1

    return cleaned, threw_out


def fetch_data():
    client = restore_session()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=180)

    print(
        f"Fetching running activities between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}..."
    )
    activities = client.get_activities_by_date(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        "running",
    )

    if not activities:
        raise RuntimeError("No running activities found in last 180 days.")

    print(f"Found {len(activities)} running activities.")

    latest_run = activities[0]
    activity_id = latest_run["activityId"]
    print(
        f"Processing Activity ID: {activity_id} ({latest_run.get('activityName')})"
    )

    details = client.get_activity_details(activity_id)

    splits = {}
    try:
        splits = client.get_activity_splits(activity_id)
    except Exception as e:
        print(f"Warning: Could not fetch lap splits: {e}")

    # Map metricDescriptors to activityDetailMetrics array positions
    descriptors = details.get("metricDescriptors", [])
    raw_metrics = details.get("activityDetailMetrics", [])

    metric_map = {d["key"]: d["metricsIndex"] for d in descriptors}

    hr_idx = metric_map.get("directHeartRate")
    speed_idx = metric_map.get("directSpeed")

    hr_stream = []
    speed_stream = []

    for row in raw_metrics:
        vals = row.get("metrics", [])
        hr_val = (
            vals[hr_idx]
            if hr_idx is not None and hr_idx < len(vals)
            else None
        )
        speed_val = (
            vals[speed_idx]
            if speed_idx is not None and speed_idx < len(vals)
            else None
        )

        hr_stream.append(hr_val)
        speed_stream.append(speed_val)

    clean_hr_stream, threw_out_count = filter_hr_spikes(hr_stream)

    paces_min_mile = []
    for s in speed_stream:
        if s and s > 0:
            pace_val = 26.8224 / s
            if pace_val < 30.0:
                paces_min_mile.append(round(pace_val, 2))
            else:
                paces_min_mile.append(None)
        else:
            paces_min_mile.append(None)

    valid_hrs = [h for h in clean_hr_stream if h is not None]
    computed_avg_hr = (
        round(sum(valid_hrs) / len(valid_hrs), 1) if valid_hrs else 0
    )
    reported_avg_hr = latest_run.get("averageHR", 0)

    output = {
        "summary": {
            "activityId": activity_id,
            "name": latest_run.get("activityName"),
            "date": latest_run.get("startTimeLocal"),
            "distance_meters": latest_run.get("distance"),
            "distance_miles": round(
                latest_run.get("distance", 0) / 1609.34, 2
            ),
            "duration_seconds": latest_run.get("duration"),
            "reported_avg_hr": reported_avg_hr,
            "reported_max_hr": latest_run.get("maxHR"),
            "reported_avg_speed_m_s": latest_run.get("averageSpeed"),
        },
        "validation": {
            "computed_avg_hr": computed_avg_hr,
            "hr_spikes_removed": threw_out_count,
        },
        "splits": splits.get("lapDTOs", []),
        "per_second_clean_hr": clean_hr_stream,
        "per_second_paces_min_mile": paces_min_mile,
        "all_historical_runs_summary": [
            {
                "activityId": act.get("activityId"),
                "date": act.get("startTimeLocal"),
                "name": act.get("activityName"),
                "distance_m": act.get("distance"),
                "duration_s": act.get("duration"),
                "avg_hr": act.get("averageHR"),
                "max_hr": act.get("maxHR"),
                "avg_speed_m_s": act.get("averageSpeed"),
            }
            for act in activities[:20]
        ],
    }

    with open("latest_run.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Successfully generated latest_run.json")


if __name__ == "__main__":
    fetch_data()