import html
import json
import os
import statistics
import time

from openai import OpenAI, RateLimitError


DEFAULT_MODEL = "google/gemma-4-31b-it:free"


def format_pace(pace):
    if pace is None or pace <= 0:
        return "-"
    minutes = int(pace)
    seconds = round((pace - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"


def format_duration(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def observed_hr_bands(hr_stream):
    values = [value for value in hr_stream if value is not None]
    if not values:
        return []
    maximum = max(values)
    lower_bound = maximum * 0.5
    bands = []
    for index in range(5):
        lower = lower_bound + (maximum - lower_bound) * index / 5
        upper = lower_bound + (maximum - lower_bound) * (index + 1) / 5
        count = sum(lower <= value < upper for value in values)
        if index == 4:
            count += sum(value == maximum for value in values)
        bands.append({"label": f"Band {index + 1}", "lower": round(lower), "upper": round(upper), "percent": round(count / len(values) * 100, 1)})
    return bands


def clean_json_response(content):
    content = content.strip()
    if "```" in content:
        content = content.split("```", 1)[1]
        content = content.removeprefix("json").strip()
        content = content.split("```", 1)[0].strip()
    return json.loads(content)


def pace_from_speed(speed):
    return 16.6667 / speed if speed and speed > 0 else None


def rolling_average(values, window):
    valid = [value for value in values if value is not None]
    if len(valid) < window:
        return max(valid) if valid else None
    return max(
        sum(valid[index:index + window]) / window
        for index in range(len(valid) - window + 1)
    )


def derive_training_evidence(run_data):
    runs = run_data.get("all_historical_runs_summary", [])
    detailed_runs = [run for run in runs if run.get("clean_hr")]
    max_candidate = None
    for run in detailed_runs:
        sustained = rolling_average(run["clean_hr"], 30)
        if sustained is not None and (max_candidate is None or sustained > max_candidate["value"]):
            max_candidate = {"value": round(sustained, 1), "activity": run.get("name"), "date": run.get("date"), "activityId": run.get("activityId")}

    threshold_candidates = []
    for run in detailed_runs:
        clean_hr = [value for value in run["clean_hr"] if value is not None]
        paces = [value for value in run.get("pace_min_km", []) if value is not None]
        duration = run.get("duration_s") or len(clean_hr)
        if duration >= 1200 and clean_hr and paces:
            threshold_candidates.append({
                "activity": run.get("name"),
                "date": run.get("date"),
                "activityId": run.get("activityId"),
                "duration_s": duration,
                "best_sustained_pace_min_km": round(statistics.mean(sorted(paces)[:max(30, len(paces) // 10)]), 2),
                "hardest_sustained_hr": round(statistics.mean(sorted(clean_hr)[-max(30, len(clean_hr) // 10):]), 1),
                "peak_clean_hr": max(clean_hr),
            })
    threshold_candidates.sort(key=lambda item: item["best_sustained_pace_min_km"])
    best_threshold = threshold_candidates[0] if threshold_candidates else None
    max_hr = max_candidate["value"] if max_candidate else None
    threshold_hr = best_threshold["hardest_sustained_hr"] if best_threshold else None
    threshold_pace = best_threshold["best_sustained_pace_min_km"] if best_threshold else None

    zone_basis = threshold_hr or (max_hr * 0.85 if max_hr else None)
    zone_definitions = []
    if zone_basis:
        bounds = [0.70, 0.80, 0.87, 0.93, 1.01]
        previous = 0
        for index, upper in enumerate(bounds):
            zone_definitions.append({"zone": index + 1, "lower_bpm": round(zone_basis * previous), "upper_bpm": round(zone_basis * upper)})
            previous = upper

    latest_hr = run_data.get("per_second_clean_hr", [])
    zone_time = []
    for zone in zone_definitions:
        count = sum(zone["lower_bpm"] <= value < zone["upper_bpm"] for value in latest_hr if value is not None)
        zone_time.append({**zone, "seconds": count, "percent": round(count / len(latest_hr) * 100, 1) if latest_hr else 0})

    activity_trends = []
    for run in runs:
        avg_hr = run.get("reported_avg_hr") or run.get("computed_avg_hr")
        pace = pace_from_speed(run.get("reported_avg_speed_m_s"))
        if threshold_hr and avg_hr:
            effort = "quality" if avg_hr >= threshold_hr * 0.95 else "steady" if avg_hr >= threshold_hr * 0.85 else "easy"
        else:
            effort = "unclassified"
        activity_trends.append({
            "activityId": run.get("activityId"),
            "effort": effort,
            "pace_min_km": round(pace, 2) if pace else None,
            "avg_hr": avg_hr,
            "distance_km": run.get("distance_km"),
        })

    return {
        "runs_with_detail": len(detailed_runs),
        "max_hr": max_candidate,
        "threshold": {
            "heart_rate": threshold_hr,
            "pace_min_km": threshold_pace,
            "basis": "hardest sustained effort" if best_threshold else "unavailable: no run with at least 20 minutes of detailed data",
            "candidate": best_threshold,
        },
        "zones": zone_time,
        "activity_trends": activity_trends,
        "validation_failures": [
            run for run in detailed_runs
            if abs(run.get("hr_difference") or 0) > 10 or abs(run.get("speed_difference_m_s") or 0) > 0.5
        ],
    }


def request_insights(client, model, run_data):
    summary = run_data["summary"]
    validation = run_data["validation"]
    historical = [
        {
            key: run.get(key)
            for key in (
                "activityId", "date", "name", "distance_km", "duration_s",
                "reported_avg_hr", "computed_avg_hr", "reported_max_hr",
                "reported_avg_speed_m_s", "hr_spikes_removed",
            )
        }
        for run in run_data.get("all_historical_runs_summary", [])
    ]
    clean_hr = [value for value in run_data.get("per_second_clean_hr", []) if value is not None]
    pace = [value for value in run_data.get("per_second_paces_min_km", []) if value is not None]
    evidence = run_data["derived_evidence"]
    prompt = f"""
You are a careful running coach. Analyze this Garmin run using only the supplied data.
Return ONLY valid JSON with exactly these string fields:
headline, effort_summary, pacing_insight, heart_rate_insight, threshold_assessment,
training_zones, zone_comparison, next_run, confidence, caveats.
Keep each value under 500 characters. Be concrete: cite numbers, identify whether
this was easy/steady/quality, mention pacing changes, and do not invent an app zone
model when none is supplied. Say that evidence is limited when appropriate.

Run summary: {json.dumps(summary)}
Validation: {json.dumps(validation)}
Clean HR range: {min(clean_hr) if clean_hr else None}-{max(clean_hr) if clean_hr else None},
clean HR average: {round(statistics.mean(clean_hr), 1) if clean_hr else None}
Pace range: {min(pace) if pace else None}-{max(pace) if pace else None} min/km
All historical running activities: {json.dumps(historical)}
Splits: {json.dumps(run_data.get("splits", [])[:30])}
Derived evidence computed locally (do not override it): {json.dumps(evidence)}
Explain max HR activity/date, threshold evidence, cross-activity trends, and
heat/negative-split limitations if data is unavailable. Never claim a paid app's
zones were identified unless supplied.
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                extra_body={"provider": {"allow_fallbacks": True}},
            )
            result = clean_json_response(response.choices[0].message.content)
            fields = [
                "headline", "effort_summary", "pacing_insight", "heart_rate_insight",
                "threshold_assessment", "training_zones", "zone_comparison", "next_run",
                "confidence", "caveats",
            ]
            return {field: str(result.get(field, "Not enough evidence.")) for field in fields}
        except RateLimitError as error:
            if attempt == 2:
                print("OpenRouter rate limit persisted; using local evidence summary.")
                break
            delay_seconds = 5 * (2**attempt)
            print(f"OpenRouter rate limit; retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
        except (ValueError, TypeError, KeyError) as error:
            print(f"Could not parse structured model response: {error}")
            break
        except Exception as error:
            print(f"Could not obtain structured coaching insights: {error}")
            break

    summary = run_data["summary"]
    return {
        "headline": f"{summary.get('name') or 'Latest run'}: evidence dashboard",
        "effort_summary": "The charts below show the recorded effort. A model-generated coaching summary was unavailable.",
        "pacing_insight": "Use the pace chart and split table to judge consistency across the run.",
        "heart_rate_insight": "Heart-rate values shown here use the cleaned per-second stream.",
        "threshold_assessment": "A single run is not enough evidence to establish threshold heart rate or threshold pace.",
        "training_zones": "The bars are provisional observed HR bands, not validated training zones.",
        "next_run": "Keep the next run easy enough to compare pace and heart rate with this effort.",
        "confidence": "Data visualization: high. Coaching interpretation: limited.",
        "caveats": "Threshold and training-zone conclusions need multiple comparable efforts, temperature, terrain, and verified effort context.",
    }


def build_dashboard(run_data, insights):
    summary = run_data["summary"]
    validation = run_data["validation"]
    hr = run_data.get("per_second_clean_hr", [])
    pace = run_data.get("per_second_paces_min_km", [])
    split_rows = run_data.get("splits", [])
    hr_bands = observed_hr_bands(hr)
    chart_data = {"hr": hr, "pace": pace}
    evidence = run_data["derived_evidence"]
    average_speed = summary.get("reported_avg_speed_m_s") or 0
    cards = [
        ("Distance", f"{summary.get('distance_km', 0):.2f} km"),
        ("Moving time", format_duration(summary.get("duration_seconds"))),
        ("Average HR", f"{summary.get('reported_avg_hr') or validation.get('computed_avg_hr') or '-'} bpm"),
        ("Avg pace", format_pace(pace_from_speed(average_speed))),
    ]
    card_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    insight_html = "".join(
        f'<article class="insight"><h3>{html.escape(title)}</h3><p>{html.escape(insights[key])}</p></article>'
        for key, title in [
            ("effort_summary", "Effort read"),
            ("pacing_insight", "Pacing"),
            ("heart_rate_insight", "Heart rate"),
            ("threshold_assessment", "Threshold evidence"),
            ("training_zones", "Training zones"),
            ("zone_comparison", "Zone comparison"),
            ("next_run", "Next run"),
        ]
    )
    split_html = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(key, '-')))}</td>"
            for key in ("splitType", "splitTypeValue", "averageSpeed", "averageHR", "elevationGain")
        ) + "</tr>"
        for row in split_rows
    ) or '<tr><td colspan="5">No lap splits were returned for this activity.</td></tr>'
    bands_html = "".join(
        f'<div class="band"><div><strong>{band["label"]}</strong><span>{band["lower"]}-{band["upper"]} bpm</span></div><b style="width:{max(band["percent"], 1)}%"></b><em>{band["percent"]}%</em></div>'
        for band in hr_bands
    ) or '<p class="muted">No heart-rate samples were returned.</p>'
    zone_html = "".join(
        f'<tr><td>Z{zone["zone"]}</td><td>{zone["lower_bpm"]}-{zone["upper_bpm"]} bpm</td><td>{format_duration(zone["seconds"])}</td><td>{zone["percent"]}%</td></tr>'
        for zone in evidence.get("zones", [])
    ) or '<tr><td colspan="4">No threshold evidence was available to derive zones.</td></tr>'
    max_hr = evidence.get("max_hr") or {}
    threshold = evidence.get("threshold") or {}
    effort_by_id = {item["activityId"]: item["effort"] for item in evidence.get("activity_trends", [])}
    history_rows = "".join(
        f'<tr><td>{html.escape(str(run.get("date") or "-")[:10])}</td><td>{html.escape(str(run.get("name") or "Run"))}</td><td>{run.get("distance_km", 0):.2f}</td><td>{format_pace(pace_from_speed(run.get("reported_avg_speed_m_s")))}</td><td>{run.get("reported_avg_hr") or "-"}</td><td>{run.get("reported_max_hr") or "-"}</td><td>{format_duration(run.get("duration_s"))}</td><td>{effort_by_id.get(run.get("activityId"), "unclassified")}</td><td>{run.get("hr_spikes_removed", 0)}</td></tr>'
        for run in run_data.get("all_historical_runs_summary", [])
    ) or '<tr><td colspan="9">No historical activity evidence was returned.</td></tr>'
    evidence_html = f'''<div class="evidence-grid"><div><span>Clean sustained max HR</span><strong>{max_hr.get("value", "-")} bpm</strong><small>{html.escape(str(max_hr.get("activity") or "No qualifying effort"))} · {html.escape(str(max_hr.get("date") or ""))}</small></div><div><span>Threshold HR</span><strong>{threshold.get("heart_rate") or "-"} bpm</strong><small>{html.escape(str(threshold.get("basis")))}</small></div><div><span>Threshold pace</span><strong>{format_pace(threshold.get("pace_min_km"))}</strong><small>Candidate from sustained efforts</small></div><div><span>Detailed runs</span><strong>{evidence.get("runs_with_detail", 0)}</strong><small>{len(evidence.get("validation_failures", []))} validation warnings</small></div></div>'''
    evidence_html += f'''<h3>All analyzed activities</h3><p class="muted">Every running activity with returned detail evidence. Pace is minutes per kilometer.</p><table><thead><tr><th>Date</th><th>Activity</th><th>Distance km</th><th>Avg pace</th><th>Avg HR</th><th>Max HR</th><th>Time</th><th>Effort</th><th>Spikes</th></tr></thead><tbody>{history_rows}</tbody></table>'''
    title = html.escape(summary.get("name") or "Latest quality run")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} | Run analysis</title><style>
:root{{--ink:#17212b;--muted:#68737d;--line:#dce3e7;--paper:#f5f7f8;--accent:#e85d3f;--teal:#117c78}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:auto;padding:28px 18px 60px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:22px}}h1{{font-size:clamp(25px,4vw,42px);line-height:1.05;margin:5px 0}}h2{{font-size:20px;margin:0 0 14px}}h3{{font-size:15px;margin:0 0 6px}}.eyebrow{{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.date,.muted{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}}.metric,.panel,.insight{{background:white;border:1px solid var(--line);border-radius:8px}}.metric{{padding:16px}}.metric span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.metric strong{{display:block;font-size:24px;margin-top:5px}}
.layout{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(280px,.65fr);gap:16px}}.panel{{padding:18px;margin-bottom:16px}}.wide{{grid-column:1/-1}}canvas{{display:block;width:100%;height:260px;border-bottom:1px solid var(--line);cursor:crosshair}}.legend{{display:flex;gap:18px;color:var(--muted);font-size:12px;margin-top:10px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--accent)}}.dot.teal{{background:var(--teal)}}
.insights{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}.insight{{padding:14px;border-radius:6px}}.insight p{{margin:0;color:#3c4850}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 8px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}.note{{border-left:3px solid var(--accent);padding-left:12px;color:var(--muted)}}
.band{{display:grid;grid-template-columns:105px 1fr 45px;gap:10px;align-items:center;margin:13px 0;font-size:12px}}.band div span{{display:block;color:var(--muted)}}.band b{{height:10px;background:var(--teal);border-radius:2px;display:block}}.band em{{font-style:normal;color:var(--muted);text-align:right}}
.evidence-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}}.evidence-grid div{{background:var(--paper);padding:12px;border-radius:6px}}.evidence-grid span,.evidence-grid small{{display:block;color:var(--muted);font-size:12px}}.evidence-grid strong{{display:block;font-size:20px;margin:4px 0}}
@media(max-width:700px){{main{{padding:20px 12px 40px}}header{{display:block}}.metrics{{grid-template-columns:repeat(2,1fr)}}.layout{{display:block}}.insights{{grid-template-columns:1fr}}.evidence-grid{{grid-template-columns:repeat(2,1fr)}}.panel{{padding:14px;overflow:auto}}table{{min-width:570px}}}}
</style></head><body><main><header><div><div class="eyebrow">Evidence-led run analysis</div><h1>{title}</h1><div class="date">{html.escape(str(summary.get('date') or 'Date unavailable'))}</div></div><div class="muted">{validation.get('hr_spikes_removed', 0)} HR spikes removed</div></header>
<section class="metrics">{card_html}</section><div class="layout"><section class="panel"><h2>Per-second effort</h2><canvas id="chart" aria-label="Interactive heart rate and pace chart"></canvas><div class="legend"><span><i class="dot"></i>Heart rate</span><span><i class="dot teal"></i>Pace</span><span id="hover" class="muted">Hover the chart for a reading</span></div></section>
<section class="panel"><h2>Coach read</h2><div class="insights">{insight_html}</div><p class="note"><strong>Confidence:</strong> {html.escape(insights['confidence'])}<br><strong>Limits:</strong> {html.escape(insights['caveats'])}</p></section>
<section class="panel"><h2>Heart-rate distribution</h2><p class="muted">Provisional observed bands from the cleaned stream, not validated physiological zones.</p>{bands_html}</section><section class="panel"><h2>Evidence behind the zones</h2>{evidence_html}<table><thead><tr><th>Zone</th><th>HR range</th><th>Time</th><th>Share</th></tr></thead><tbody>{zone_html}</tbody></table><p class="note">Derived from cleaned per-second data. These are evidence-based estimates, not medical guidance. App-specific zones require the app's configured basis.</p></section><section class="panel wide"><h2>Splits and laps</h2><table><thead><tr><th>Type</th><th>Value</th><th>Avg speed</th><th>Avg HR</th><th>Elevation gain</th></tr></thead><tbody>{split_html}</tbody></table></section></div></main>
<script>const data={json.dumps(chart_data,separators=(',', ':'))};const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),hover=document.getElementById('hover');function draw(){{const d=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*d;canvas.height=h*d;ctx.scale(d,d);ctx.clearRect(0,0,w,h);const series=[['hr','#e85d3f',data.hr],['pace','#117c78',data.pace]],all=series.flatMap(x=>x[2].filter(v=>v!=null));if(!all.length)return;const min=Math.min(...all),max=Math.max(...all);series.forEach(([name,color,values])=>{{ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;values.forEach((v,i)=>{{if(v==null)return;const x=i/(values.length-1)*w,y=h-(v-min)/(max-min||1)*(h-20)-10;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke()}});canvas.onmousemove=e=>{{const i=Math.min(data.hr.length-1,Math.max(0,Math.round((e.offsetX/w)*(data.hr.length-1))));hover.textContent=`${{i}}s · HR ${{data.hr[i]??'-'}} bpm · Pace ${{data.pace[i]??'-'}} min/km`}}}};addEventListener('resize',draw);draw();</script></body></html>'''


def main():
    with open("latest_run.json", "r") as file:
        run_data = json.load(file)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    run_data["derived_evidence"] = derive_training_evidence(run_data)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        default_headers={"Authorization": f"Bearer {key}"},
    )
    model = os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    insights = request_insights(client, model, run_data)
    with open("index.html", "w") as file:
        file.write(build_dashboard(run_data, insights))
    print("Generated interactive index.html dashboard.")


if __name__ == "__main__":
    main()
