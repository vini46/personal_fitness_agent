import os
import json
import time
from openai import OpenAI, RateLimitError

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not openrouter_api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it as a GitHub Actions repository secret."
    )
if not openrouter_api_key.startswith("sk-or-"):
    raise RuntimeError(
        "OPENROUTER_API_KEY is not an OpenRouter key. Create a key at openrouter.ai/keys."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    default_headers={"Authorization": f"Bearer {openrouter_api_key}"},
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

model = os.environ.get("OPENROUTER_MODEL", "").strip() or "google/gemma-4-31b-it:free"
response = None
for attempt in range(3):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"provider": {"allow_fallbacks": True}},
        )
        break
    except RateLimitError as error:
        if attempt == 2:
            raise RuntimeError(
                f"OpenRouter rate limit persisted for model {model}. "
                "Set OPENROUTER_MODEL to another available model or retry later."
            ) from error

        delay_seconds = 5 * (2**attempt)
        print(
            f"OpenRouter rate limit for {model}; retrying in "
            f"{delay_seconds} seconds..."
        )
        time.sleep(delay_seconds)

html_report = response.choices[0].message.content

# Strip markdown backticks if returned
if "```html" in html_report:
    html_report = html_report.split("```html")[1].split("```")[0]
elif "```" in html_report:
    html_report = html_report.split("```")[1].split("```")[0]

with open("index.html", "w") as f:
    f.write(html_report)

print("Generated index.html dashboard.")