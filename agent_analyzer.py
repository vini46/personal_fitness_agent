import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

with open("latest_run.json", "r") as f:
    run_data = json.load(f)

prompt = f"""
You are an expert running performance analyst. Rebuild the run analysis for this athlete's activity.

Here is the preprocessed activity data (HR clean-up applied, stream mapped):
{json.dumps(run_data, indent=2)}

Perform the following:
1. Work out REAL training zones from the clean per-second HR stream and efforts.
2. Determine Max HR and Threshold HR/Pace.
3. Compute time-in-zone percentage from per-second data.
4. Output a clean, single-file HTML dashboard with a light background and mobile layout containing:
   - Distance, time, pace, splits, lap table, and time-in-zone horizontal bars.
   - Contrast between these derived zones vs typical app default zone assumptions.
   - State clearly the exact seconds-per-mile difference between the real easy pace ceiling and typical default zone model ceilings.
"""

response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-001:free",  # Free model on OpenRouter
    messages=[{"role": "user", "content": prompt}],
)

html_report = response.choices[0].message.content

# Strip markdown backticks if returned
if "```html" in html_report:
    html_report = html_report.split("```html")[1].split("```")[0]
elif "```" in html_report:
    html_report = html_report.split("```")[1].split("```")[0]

with open("index.html", "w") as f:
    f.write(html_report)

print("Generated index.html dashboard.")