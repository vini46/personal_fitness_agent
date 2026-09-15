import html
import json
import math
import statistics
from datetime import datetime, timedelta


BASELINE_SLEEP_HOURS = 8.0
STAGE_NAMES = {0: "Deep", 1: "Light", 2: "REM", 3: "Awake"}


def first_value(value, keys):
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)):
                return candidate
        for child in value.values():
            result = first_value(child, keys)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = first_value(child, keys)
            if result is not None:
                return result
    return None


def clamp(value, lower=0, upper=100):
    return max(lower, min(upper, value))


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 2) if values else None


def std_or_none(values):
    values = [value for value in values if value is not None]
    return round(statistics.stdev(values), 2) if len(values) > 1 else None


def sleep_record(record):
    payload = record.get("sleep") or {}
    dto = payload.get("dailySleepDTO", payload if isinstance(payload, dict) else {})
    levels = payload.get("sleepLevels", []) if isinstance(payload, dict) else []
    stage_seconds = {name: 0 for name in STAGE_NAMES.values()}
    for level in levels or []:
        if not isinstance(level, dict):
            continue
        stage = level.get("activityLevel", level.get("stage", level.get("level")))
        try:
            stage = int(stage)
        except (TypeError, ValueError):
            continue
        if stage not in STAGE_NAMES:
            continue
        start = level.get("startGMT", level.get("startTimeGMT", level.get("start")))
        end = level.get("endGMT", level.get("endTimeGMT", level.get("end")))
        seconds = level.get("seconds")
        if seconds is None and isinstance(start, str) and isinstance(end, str):
            try:
                seconds = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
            except ValueError:
                seconds = 0
        stage_seconds[STAGE_NAMES[stage]] += max(0, int(seconds or 0))
    slept = first_value(dto, ["totalSleepSeconds", "sleepTimeSeconds", "sleepSeconds"])
    deep = first_value(dto, ["deepSleepSeconds"])
    light = first_value(dto, ["lightSleepSeconds"])
    rem = first_value(dto, ["remSleepSeconds"])
    awake = first_value(dto, ["awakeSleepSeconds"])
    reported = {"Deep": deep, "Light": light, "REM": rem, "Awake": awake}
    if not levels and any(value is not None for value in reported.values()):
        stage_seconds = {key: int(value or 0) for key, value in reported.items()}
    if slept is None:
        slept = sum(stage_seconds[key] for key in ("Deep", "Light", "REM"))
    in_bed = first_value(dto, ["timeInBedSeconds", "totalTimeInBedSeconds"]) or slept + stage_seconds["Awake"]
    stage_total = sum(stage_seconds.values())
    reported_total = sum(value or 0 for value in reported.values())
    return {
        "date": record["date"],
        "slept_seconds": int(slept or 0),
        "in_bed_seconds": int(in_bed or 0),
        "stages": stage_seconds,
        "reported_stages": reported,
        "reconciles": abs(stage_total - reported_total) <= 60 if levels and reported_total else None,
        "reconcile_difference_seconds": stage_total - reported_total if levels and reported_total else None,
        "efficiency": round(slept / in_bed * 100, 1) if in_bed else None,
    }


def hrv_value(record):
    return first_value(record.get("hrv"), ["lastNightAvg", "lastNightAverage", "hrvValue", "weeklyAvg"])


def rhr_value(record):
    return first_value(record.get("resting_hr"), ["restingHeartRate", "restingHr", "allDayAvg", "value"])


def trimp_for_run(run, hr_rest, hr_max):
    stream = [value for value in run.get("clean_hr", []) if value is not None]
    if not stream or not hr_rest or not hr_max or hr_max <= hr_rest:
        return 0.0
    total = 0.0
    for index in range(0, len(stream), 60):
        hr_ratio = max(0, min(1, (statistics.mean(stream[index:index + 60]) - hr_rest) / (hr_max - hr_rest)))
        total += hr_ratio * 0.64 * math.exp(1.92 * hr_ratio)
    return round(total, 2)


def build_data():
    with open("wellness_data.json") as file:
        raw = json.load(file)
    with open("latest_run.json") as file:
        run_data = json.load(file)
    days = raw.get("days", [])
    sleeps = [sleep_record(day) for day in days]
    hrv_values = [hrv_value(day) for day in days]
    rhr_values = [rhr_value(day) for day in days]
    valid_hrv = [value for value in hrv_values if value is not None]
    valid_rhr = [value for value in rhr_values if value is not None]
    sleep_values = [item["slept_seconds"] for item in sleeps if item["slept_seconds"] >= 4 * 3600]
    hrv_mean, hrv_sd = mean_or_none(valid_hrv), std_or_none(valid_hrv)
    rhr_30 = valid_rhr[:30]
    rhr_14 = valid_rhr[:14]
    rhr_mean, rhr_sd = mean_or_none(rhr_30), std_or_none(rhr_30)
    sleep_median = statistics.median(sleep_values) if sleep_values else 0
    sleep_p75 = sorted(sleep_values)[max(0, math.ceil(len(sleep_values) * 0.75) - 1)] if sleep_values else 0
    hr_streams = [value for run in run_data.get("all_historical_runs_summary", []) for value in run.get("clean_hr", []) if value is not None]
    hr_max = max(hr_streams) if hr_streams else None
    hr_rest = rhr_mean or 60
    activity_loads = []
    for run in run_data.get("all_historical_runs_summary", []):
        load = trimp_for_run(run, hr_rest, hr_max)
        activity_loads.append({"date": str(run.get("date", ""))[:10], "name": run.get("name") or "Run", "load": load, "duration_s": run.get("duration_s") or 0, "avg_hr": run.get("reported_avg_hr"), "has_hr": bool(run.get("clean_hr"))})
    by_date = {}
    for activity in activity_loads:
        by_date[activity["date"]] = by_date.get(activity["date"], 0) + activity["load"]
    latest = sleeps[0] if sleeps else {"slept_seconds": 0, "in_bed_seconds": 0, "stages": {}, "efficiency": None}
    yesterday = by_date.get((datetime.fromisoformat(latest["date"]) - timedelta(days=1)).strftime("%Y-%m-%d"), 0) if latest.get("date") else 0
    shortfall = sum(max(0, BASELINE_SLEEP_HOURS * 3600 - item["slept_seconds"]) for item in sleeps[1:4])
    debt_hours = min(0.35 * shortfall / 3600, 1.5)
    strain_yesterday = 21 * math.log(1 + (yesterday + 4) / 12) / math.log(1 + 90 / 12)
    strain_addition = 1.7 / (1 + math.exp((17 - strain_yesterday) / 3.5))
    naps = sum(max(0, BASELINE_SLEEP_HOURS * 3600 - item["slept_seconds"]) for item in sleeps[:1] if item["slept_seconds"] < 4 * 3600) / 3600
    sleep_need = BASELINE_SLEEP_HOURS + strain_addition + debt_hours - naps
    last_hrv = valid_hrv[0] if valid_hrv else None
    today_rhr = valid_rhr[0] if valid_rhr else None
    hrv_score = clamp(50 + 20 * ((last_hrv - hrv_mean) / hrv_sd)) if last_hrv is not None and hrv_mean is not None and hrv_sd else None
    rhr_score = clamp(50 - 20 * ((today_rhr - rhr_mean) / rhr_sd)) if today_rhr is not None and rhr_mean is not None and rhr_sd else None
    sleep_performance = min(100, latest["slept_seconds"] / (sleep_need * 3600) * 100) if sleep_need else 0
    components = [(hrv_score, 0.50), (rhr_score, 0.25), (sleep_performance, 0.25)]
    available = [(score, weight) for score, weight in components if score is not None]
    weight_total = sum(weight for _, weight in available) or 1
    recovery = round(sum(score * weight / weight_total for score, weight in available), 1)
    recovery_band = "green" if recovery >= 67 else "yellow" if recovery >= 34 else "red"
    today_load = by_date.get(latest.get("date"), 0) + 4
    today_strain = round(21 * math.log(1 + today_load / 12) / math.log(1 + 90 / 12), 1)
    target = {"green": "14.0-18.0", "yellow": "9.0-13.5", "red": "0-8.0"}[recovery_band]
    return {"latest_sleep": latest, "sleep_history": sleeps[:7], "hrv": {"last": last_hrv, "mean": hrv_mean, "sd": hrv_sd, "min": min(valid_hrv) if valid_hrv else None, "max": max(valid_hrv) if valid_hrv else None, "count": len(valid_hrv)}, "rhr": {"today": today_rhr, "mean_30": rhr_mean, "sd_30": rhr_sd, "mean_14": mean_or_none(rhr_14), "count": len(valid_rhr)}, "sleep_stats": {"median_seconds": sleep_median, "p75_seconds": sleep_p75, "count": len(sleep_values)}, "scores": {"recovery": recovery, "band": recovery_band, "hrv_score": hrv_score, "rhr_score": rhr_score, "sleep_score": round(sleep_performance, 1), "weights": {"HRV": round(0.50 / weight_total, 2) if hrv_score is not None else 0, "RHR": round(0.25 / weight_total, 2) if rhr_score is not None else 0, "Sleep": round(0.25 / weight_total, 2)}, "strain": today_strain, "target": target}, "sleep_need": {"baseline_hours": BASELINE_SLEEP_HOURS, "strain_hours": round(strain_addition, 2), "debt_hours": round(debt_hours, 2), "naps_hours": round(naps, 2), "need_hours": round(sleep_need, 2)}, "activities": activity_loads[:30], "strain_history": [{"date": date, "strain": round(21 * math.log(1 + (load + 4) / 12) / math.log(1 + 90 / 12), 1)} for date, load in list(by_date.items())[:7]], "quality": {"stage_checks": [item for item in sleeps if item.get("reconciles") is not None][:3], "deep_percent_range": [round(min((item["stages"].get("Deep", 0) / item["slept_seconds"] * 100) for item in sleeps if item["slept_seconds"] > 0), 1) if any(item["slept_seconds"] > 0 for item in sleeps) else None, round(max((item["stages"].get("Deep", 0) / item["slept_seconds"] * 100) for item in sleeps if item["slept_seconds"] > 0), 1) if any(item["slept_seconds"] > 0 for item in sleeps) else None], "rem_percent_range": [round(min((item["stages"].get("REM", 0) / item["slept_seconds"] * 100) for item in sleeps if item["slept_seconds"] > 0), 1) if any(item["slept_seconds"] > 0 for item in sleeps) else None, round(max((item["stages"].get("REM", 0) / item["slept_seconds"] * 100) for item in sleeps if item["slept_seconds"] > 0), 1) if any(item["slept_seconds"] > 0 for item in sleeps) else None], "missing_hr_activities": [item for item in activity_loads if not item["has_hr"]]}}


def esc(value):
    return html.escape(str(value if value is not None else "-"))


def hours(seconds):
    return f"{int(seconds or 0) // 3600}h {int(seconds or 0) % 3600 // 60:02d}m"


def page(data):
    score = data["scores"]
    today_load = round((math.exp(score["strain"] * math.log(1 + 90 / 12) / 21) - 1) * 12, 1)
    recovery_value = score["recovery"]
    band_color = {"green": "#16866e", "yellow": "#c78a21", "red": "#c7524a"}[score["band"]]
    stage = data["latest_sleep"]["stages"]
    stage_total = sum(stage.values()) or 1
    stage_rows = "".join(f'<div class="stage-row"><span>{name}</span><b><i style="width:{value / stage_total * 100:.1f}%"></i></b><strong>{hours(value)}</strong></div>' for name, value in stage.items())
    activity_rows = "".join(f'<tr><td>{esc(item["date"])}</td><td>{esc(item["name"])}</td><td>{item["load"]:.1f}</td><td>{hours(item["duration_s"])}</td><td>{esc(item["avg_hr"])}</td><td>{"Recorded" if item["has_hr"] else "No HR data"}</td></tr>' for item in data["activities"])
    strain_rows = "".join(f'<div class="strain-bar"><span>{esc(item["date"])}</span><i style="height:{max(8, item["strain"] / 21 * 150):.0f}px"></i><strong>{item["strain"]}</strong></div>' for item in data["strain_history"])
    sleep_strip = "".join(f'<div class="sleep-day"><strong>{esc(item["date"])[5:]}</strong><b style="height:{max(8, item["slept_seconds"] / (10 * 3600) * 110):.0f}px"></b><span>{hours(item["slept_seconds"])}</span></div>' for item in data["sleep_history"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery, strain and sleep</title><style>
:root{{--ink:#17212b;--muted:#6a747b;--line:#dfe6e9;--paper:#f4f7f7;--teal:#16866e;--orange:#df7950;--blue:#4d80a8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1120px;margin:auto;padding:28px 18px 70px}}header{{display:flex;justify-content:space-between;align-items:end;gap:20px;border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:18px}}h1{{font-size:clamp(28px,5vw,48px);line-height:1.02;margin:4px 0}}h2{{font-size:21px;margin:0 0 12px}}h3{{font-size:15px;margin:0 0 8px}}.eyebrow{{color:var(--orange);font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}.muted,small{{color:var(--muted)}}.back{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:white;border-radius:999px;padding:8px 13px;font-size:13px}}.tabs{{display:flex;gap:8px;overflow-x:auto;margin-bottom:18px}}.tabs button{{border:1px solid var(--line);background:white;color:var(--muted);padding:10px 17px;border-radius:999px;cursor:pointer;font-weight:700}}.tabs button.active{{background:var(--ink);border-color:var(--ink);color:white}}.view{{display:none}}.view.active{{display:block}}.hero{{display:grid;grid-template-columns:260px 1fr;gap:22px;align-items:center;margin-bottom:16px}}.ring{{width:220px;height:220px;border-radius:50%;display:grid;place-items:center;background:conic-gradient({band_color} {recovery_value}%,#e3e9e9 0);position:relative}}.ring:after{{content:"";position:absolute;inset:16px;background:white;border-radius:50%}}.ring-content{{position:relative;z-index:1;text-align:center}}.ring strong{{display:block;font-size:52px;line-height:1}}.ring span{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.hero-copy{{background:white;border:1px solid var(--line);border-radius:10px;padding:22px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card,.panel{{background:white;border:1px solid var(--line);border-radius:9px;padding:18px;margin-bottom:14px}}.card span,.metric-label{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.card strong{{display:block;font-size:27px;margin-top:5px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.input-row{{display:grid;grid-template-columns:1fr 80px 90px 70px;gap:10px;align-items:center;border-top:1px solid var(--line);padding:12px 0}}.input-row:first-of-type{{border-top:0}}.bar{{height:9px;background:#e7eded;border-radius:5px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--teal)}}.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#17212b;color:#e8f1ef;border-radius:7px;padding:14px;white-space:pre-wrap;font-size:12px;line-height:1.7}}.stage-row{{display:grid;grid-template-columns:65px 1fr 80px;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line)}}.stage-row b{{height:12px;background:#e7eded;border-radius:5px;overflow:hidden}}.stage-row i{{display:block;height:100%;background:var(--blue)}}.stage-row strong{{font-size:13px;text-align:right}}.strain-chart,.sleep-strip{{display:flex;align-items:end;gap:14px;min-height:180px;padding:12px 4px;border-bottom:1px solid var(--line)}}.strain-bar,.sleep-day{{display:flex;flex:1;min-width:42px;height:170px;flex-direction:column;justify-content:end;align-items:center;gap:5px;font-size:11px}}.strain-bar i{{display:block;width:28px;background:var(--orange);border-radius:5px 5px 0 0}}.sleep-day b{{display:block;width:28px;background:var(--blue);border-radius:5px 5px 0 0}}.sleep-day span{{white-space:nowrap;color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 8px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}.table-wrap{{overflow-x:auto}}.warning{{border-left:3px solid var(--orange);padding-left:12px;color:var(--muted)}}@media(max-width:760px){{main{{padding:20px 12px 50px}}header{{align-items:start;display:block}}.back{{display:inline-block;margin-top:15px}}.hero{{grid-template-columns:1fr;justify-items:center}}.hero-copy{{width:100%}}.grid,.two{{grid-template-columns:1fr}}.input-row{{grid-template-columns:1fr 65px 75px 55px;font-size:13px}}}}
</style></head><body><main><header><div><div class="eyebrow">Personal performance lab</div><h1>Recovery, strain and sleep</h1><p class="muted">Local Garmin evidence, transparent formulas, no population score.</p></div><a class="back" href="run-analysis.html">Run analysis</a></header><nav class="tabs"><button class="active" data-view="recovery">Recovery</button><button data-view="strain">Strain</button><button data-view="sleep">Sleep</button></nav>
<section id="recovery" class="view active"><div class="hero"><div class="ring"><div class="ring-content"><strong>{recovery_value:.0f}</strong><span>{score["band"]} recovery</span></div></div><div class="hero-copy"><div class="eyebrow">Today</div><h2>What the score says</h2><p>Your suggested strain target is <strong>{score["target"]}</strong>. The score uses your own HRV, resting HR and sleep baselines. {"HRV baseline is thin: fewer than 14 nights." if data["hrv"]["count"] < 14 else "The HRV baseline has at least 14 nights."}</p><p class="warning">This is a transparent training signal, not medical advice.</p></div></div><div class="grid"><div class="card"><span>HRV score</span><strong>{esc(score["hrv_score"])}</strong><small>{esc(data["hrv"]["last"])} ms vs {esc(data["hrv"]["mean"])} mean</small></div><div class="card"><span>RHR score</span><strong>{esc(score["rhr_score"])}</strong><small>{esc(data["rhr"]["today"])} bpm vs {esc(data["rhr"]["mean_30"])} 30-day mean</small></div><div class="card"><span>Sleep score</span><strong>{score["sleep_score"]:.0f}</strong><small>{hours(data["latest_sleep"]["slept_seconds"])} slept</small></div></div><div class="panel"><h2>Inputs and baselines</h2><div class="input-row"><strong>HRV</strong><span>{esc(data["hrv"]["last"])} ms</span><span>mean {esc(data["hrv"]["mean"])}</span><span>{score["weights"]["HRV"] * 100:.0f}%</span></div><div class="input-row"><strong>Resting HR</strong><span>{esc(data["rhr"]["today"])} bpm</span><span>mean {esc(data["rhr"]["mean_30"])}</span><span>{score["weights"]["RHR"] * 100:.0f}%</span></div><div class="input-row"><strong>Sleep performance</strong><span>{score["sleep_score"]:.0f}%</span><span>need {data["sleep_need"]["need_hours"]:.1f}h</span><span>{score["weights"]["Sleep"] * 100:.0f}%</span></div></div><div class="two"><div class="panel"><h2>Seven-day strain</h2><div class="strain-chart">{strain_rows or '<span class="muted">No activity load data.</span>'}</div></div><div class="panel"><h2>Formula</h2><div class="formula">recovery = 0.50 * hrv_score + 0.25 * rhr_score + 0.25 * sleep_score
Green 67-100 | Yellow 34-66 | Red 0-33
Missing inputs are redistributed across available inputs.</div></div></div></section>
<section id="strain" class="view"><div class="grid"><div class="card"><span>Today's strain</span><strong>{score["strain"]:.1f}<small> / 21</small></strong><small>Target: {score["target"]}</small></div><div class="card"><span>Today's load</span><strong>{today_load:.1f}</strong><small>Includes 4.0 living load</small></div><div class="card"><span>Scale</span><strong>0-21</strong><small>TRIMPexp + logarithmic mapping</small></div></div><div class="panel"><h2>Seven-day strain</h2><div class="strain-chart">{strain_rows or '<span class="muted">No activity load data.</span>'}</div><div class="formula">day_load = exercise_load + 4.0
strain = 21 * ln(1 + day_load / 12) / ln(1 + 90 / 12)</div></div><div class="panel"><h2>Activity load ledger</h2><div class="table-wrap"><table><thead><tr><th>Date</th><th>Activity</th><th>Load points</th><th>Duration</th><th>Avg HR</th><th>HR signal</th></tr></thead><tbody>{activity_rows or '<tr><td colspan="6">No activities found.</td></tr>'}</tbody></table></div><p class="warning">Activities without HR data contribute zero load and are listed explicitly.</p></div></section>
<section id="sleep" class="view"><div class="grid"><div class="card"><span>Sleep performance</span><strong>{score["sleep_score"]:.0f}%</strong><small>{hours(data["latest_sleep"]["slept_seconds"])} of {data["sleep_need"]["need_hours"]:.1f}h needed</small></div><div class="card"><span>Time in bed</span><strong>{hours(data["latest_sleep"]["in_bed_seconds"])}</strong><small>{esc(data["latest_sleep"]["efficiency"])}% efficiency</small></div><div class="card"><span>Sleep sample</span><strong>{data["sleep_stats"]["count"]}</strong><small>valid nights over 90 days</small></div></div><div class="two"><div class="panel"><h2>Where sleep need came from</h2><div class="input-row"><strong>Baseline</strong><span>+{data["sleep_need"]["baseline_hours"]:.1f}h</span><span>adult midpoint</span></div><div class="input-row"><strong>Strain add-on</strong><span>+{data["sleep_need"]["strain_hours"]:.2f}h</span><span>yesterday {score["strain"]:.1f}</span></div><div class="input-row"><strong>Sleep debt</strong><span>+{data["sleep_need"]["debt_hours"]:.2f}h</span><span>3-night window</span></div><div class="input-row"><strong>Naps</strong><span>-{data["sleep_need"]["naps_hours"]:.2f}h</span><span>under-4h nights</span></div><div class="formula">sleep_need = baseline + f(strain) + debt - naps
sleep_performance = slept / sleep_need</div></div><div class="panel"><h2>Last night stages</h2>{stage_rows}<p class="muted">Stage mapping: 0 deep, 1 light, 2 REM, 3 awake. Stage totals reconcile within 60 seconds when raw levels exist.</p></div></div><div class="panel"><h2>Seven-night sleep strip</h2><div class="sleep-strip">{sleep_strip or '<span class="muted">No sleep data.</span>'}</div></div><div class="panel"><h2>Honesty checks</h2><p>HRV samples: <strong>{data["hrv"]["count"]}</strong>. Resting HR samples: <strong>{data["rhr"]["count"]}</strong>. Sleep nights: <strong>{data["sleep_stats"]["count"]}</strong>.</p><p>Deep range: <strong>{data["quality"]["deep_percent_range"]}</strong>% of sleep. REM range: <strong>{data["quality"]["rem_percent_range"]}</strong>% of sleep.</p><p class="warning">Sleep staging is inferred by a wrist device. Missing respiratory rate, SpO2 and skin temperature are not silently substituted.</p></div></section></main><script>document.querySelectorAll('.tabs button').forEach(button=>button.onclick=()=>{{document.querySelectorAll('.tabs button,.view').forEach(item=>item.classList.remove('active'));button.classList.add('active');document.getElementById(button.dataset.view).classList.add('active')}});</script></body></html>'''


def main():
    data = build_data()
    with open("recovery_data.json", "w") as file:
        json.dump(data, file, indent=2)
    with open("recovery.html", "w") as file:
        file.write(page(data))
    print("Generated recovery.html")


if __name__ == "__main__":
    main()
