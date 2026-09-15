import base64
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from garminconnect import Garmin


def restore_session() -> Garmin:
    """Restore a Garmin session from cached tokens or credentials."""
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

    # Garmin owns the active client; load tokens through its public login API.
    if token_dir and os.path.exists(token_dir):
        try:
            api = Garmin(email, password)
            api.login(tokenstore=token_dir)
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


def extract_activity_evidence(details: dict, summary: dict, splits: dict) -> dict:
    """Map detail arrays using Garmin descriptors and return cleaned evidence."""
    descriptors = details.get("metricDescriptors", [])
    raw_metrics = details.get("activityDetailMetrics", [])
    metric_map = {
        descriptor["key"]: descriptor["metricsIndex"]
        for descriptor in descriptors
        if "key" in descriptor and "metricsIndex" in descriptor
    }
    hr_idx = metric_map.get("directHeartRate")
    speed_idx = metric_map.get("directSpeed")

    hr_stream = []
    speed_stream = []
    for row in raw_metrics:
        values = row.get("metrics", [])
        hr_stream.append(values[hr_idx] if hr_idx is not None and hr_idx < len(values) else None)
        speed_stream.append(values[speed_idx] if speed_idx is not None and speed_idx < len(values) else None)

    clean_hr_stream, spikes_removed = filter_hr_spikes(hr_stream)
    paces = [
        round(26.8224 / speed, 2) if speed and speed > 0 and 26.8224 / speed < 30 else None
        for speed in speed_stream
    ]
    valid_hr = [value for value in clean_hr_stream if value is not None]
    valid_speed = [value for value in speed_stream if value is not None and value > 0]
    computed_avg_hr = round(sum(valid_hr) / len(valid_hr), 1) if valid_hr else None
    computed_avg_speed = round(sum(valid_speed) / len(valid_speed), 3) if valid_speed else None
    reported_avg_hr = summary.get("averageHR")
    reported_avg_speed = summary.get("averageSpeed")

    return {
        "activityId": summary.get("activityId"),
        "date": summary.get("startTimeLocal"),
        "name": summary.get("activityName"),
        "distance_m": summary.get("distance"),
        "duration_s": summary.get("duration"),
        "reported_avg_hr": reported_avg_hr,
        "computed_avg_hr": computed_avg_hr,
        "hr_difference": round(computed_avg_hr - reported_avg_hr, 1) if computed_avg_hr is not None and reported_avg_hr else None,
        "reported_avg_speed_m_s": reported_avg_speed,
        "computed_avg_speed_m_s": computed_avg_speed,
        "speed_difference_m_s": round(computed_avg_speed - reported_avg_speed, 3) if computed_avg_speed is not None and reported_avg_speed else None,
        "reported_max_hr": summary.get("maxHR"),
        "hr_spikes_removed": spikes_removed,
        "metric_map": metric_map,
        "clean_hr": clean_hr_stream,
        "pace_min_mile": paces,
        "splits": splits.get("lapDTOs", []),
    }


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

    latest_evidence = extract_activity_evidence(details, latest_run, splits)
    historical_evidence = []
    for activity in activities[:20]:
        if activity.get("activityId") == activity_id:
            historical_evidence.append(latest_evidence)
            continue
        try:
            activity_details = client.get_activity_details(activity["activityId"])
            try:
                activity_splits = client.get_activity_splits(activity["activityId"])
            except Exception as split_error:
                print(f"Warning: Could not fetch splits for {activity.get('activityId')}: {split_error}")
                activity_splits = {}
            historical_evidence.append(extract_activity_evidence(activity_details, activity, activity_splits))
        except Exception as e:
            print(f"Warning: Could not fetch detail evidence for {activity.get('activityId')}: {e}")

    clean_hr_stream = latest_evidence["clean_hr"]
    paces_min_mile = latest_evidence["pace_min_mile"]
    computed_avg_hr = latest_evidence["computed_avg_hr"] or 0
    threw_out_count = latest_evidence["hr_spikes_removed"]
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
            "descriptor_map": latest_evidence["metric_map"],
            "reported_avg_speed_m_s": latest_evidence["reported_avg_speed_m_s"],
            "computed_avg_speed_m_s": latest_evidence["computed_avg_speed_m_s"],
            "hr_difference": latest_evidence["hr_difference"],
            "speed_difference_m_s": latest_evidence["speed_difference_m_s"],
        },
        "splits": splits.get("lapDTOs", []),
        "per_second_clean_hr": clean_hr_stream,
        "per_second_paces_min_mile": paces_min_mile,
        "all_historical_runs_summary": historical_evidence,
    }

    with open("latest_run.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Successfully generated latest_run.json")


if __name__ == "__main__":
    fetch_data()