import base64
import io
import json
import os
import tarfile
from datetime import datetime, timedelta, timezone
from garminconnect import Garmin


def restore_session() -> Garmin:
    """Restores the Garmin session using the GARMIN_TOKENS_BASE64 env secret."""
    b64_tokens = os.environ.get("GARMIN_TOKENS_BASE64")
    if not b64_tokens:
        raise ValueError("GARMIN_TOKENS_BASE64 environment secret is missing!")

    # Unpack tar.gz bundle back into home directory (~/.garminconnect)
    compressed_data = base64.b64decode(b64_tokens)
    buf = io.BytesIO(compressed_data)

    home_dir = os.path.expanduser("~")
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=home_dir)

    token_path = os.path.expanduser("~/.garminconnect")
    print(f"Restored session tokens to {token_path}")

    # Resume session using stored tokens without credentials
    client = Garmin()
    client.login(token_path)
    return client


def filter_hr_spikes(hr_stream: list, threshold: int = 15) -> tuple[list, int]:
    """Filters single-sample optical wrist heart rate artifacts.

    Drops samples that jump more than threshold (15 bpm) from their immediate neighbours.
    """
    if not hr_stream:
        return [], 0

    cleaned = list(hr_stream)
    threw_out = 0

    for i in range(1, len(cleaned) - 1):
        prev_val = cleaned[i - 1]
        curr_val = cleaned[i]
        next_val = cleaned[i + 1]

        if prev_val is not None and curr_val is not None and next_val is not None:
            # Check if current sample is an isolated spike above both neighbors
            if (curr_val - prev_val > threshold) and (
                curr_val - next_val > threshold
            ):
                # Interpolate using surrounding samples
                cleaned[i] = round((prev_val + next_val) / 2)
                threw_out += 1

    return cleaned, threw_out


def fetch_data():
    client = restore_session()

    # 1. Pull activities from last 180 days
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
        raise RuntimeError(
            "No running activities found in the last 180 days."
        )

    print(f"Found {len(activities)} running activities.")

    # Select the most recent quality run
    latest_run = activities[0]
    activity_id = latest_run["activityId"]
    print(
        f"Processing Activity ID: {activity_id} ({latest_run.get('activityName')})"
    )

    # Fetch detailed per-second metrics stream and lap/split breakdowns
    details = client.get_activity_details(activity_id)

    splits = {}
    try:
        splits = client.get_activity_splits(activity_id)
    except Exception as e:
        print(
            f"Warning: Could not fetch split details: {e}. Falling back to activity summary."
        )

    # Step 3 Check: Map metricDescriptors to activityDetailMetrics array positions
    descriptors = details.get("metricDescriptors", [])
    raw_metrics = details.get("activityDetailMetrics", [])

    metric_map = {d["key"]: d["metricsIndex"] for d in descriptors}

    hr_idx = metric_map.get("directHeartRate")
    speed_idx = metric_map.get("directSpeed")
    timestamp_idx = metric_map.get("directTimestamp")

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

    # Step 4: Filter HR Spikes
    clean_hr_stream, threw_out_count = filter_hr_spikes(hr_stream)

    # Step 4: Convert speed (m/s) to pace (minutes per mile)
    # Pace (min/mi) = 26.8224 / speed_in_m_per_s
    paces_min_mile = []
    for s in speed_stream:
        if s and s > 0:
            pace_val = 26.8224 / s
            # Cap unrealistic values (e.g., standing still or GPS jitter)
            if pace_val < 30.0:  # 30:00 min/mile max
                paces_min_mile.append(round(pace_val, 2))
            else:
                paces_min_mile.append(None)
        else:
            paces_min_mile.append(None)

    # Validation step: Compare clean average HR with activity summary average
    valid_hrs = [h for h in clean_hr_stream if h is not None]
    computed_avg_hr = (
        round(sum(valid_hrs) / len(valid_hrs), 1) if valid_hrs else 0
    )
    reported_avg_hr = latest_run.get("averageHR", 0)

    print(f"Validation Check:")
    print(f" - Computed Stream Avg HR: {computed_avg_hr} bpm")
    print(f" - Reported Activity Avg HR: {reported_avg_hr} bpm")
    print(f" - Optical HR Spikes Removed: {threw_out_count}")

    # Build final output structure for OpenRouter agent
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
            for act in activities[:20]  # Pass summary of recent 20 runs for zone derivation
        ],
    }

    with open("latest_run.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Successfully created latest_run.json")


if __name__ == "__main__":
    fetch_data()