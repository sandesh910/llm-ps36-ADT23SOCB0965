"""Starts the app, runs 5 test questions in Microsoft Edge (Playwright) and saves screenshots."""
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)
PORT = 8510
URL = f"http://localhost:{PORT}"

TESTS = [
    ("2611.18747v1.pdf", "Method", "1_method.png"),
    ("2611.18747v1.pdf", "Dataset", "2_dataset.png"),  # same paper: reuses the cached index
    ("2611.18642v1.pdf", "Results", "3_results.png"),
    ("2611.18793v1.pdf", "Limitations", "4_limitations.png"),
    ("2611.18714v1.pdf", "What top-1 accuracy did the proposed model reach on ImageNet?", "5_not_in_paper.png"),
]

server = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
     "--server.port", str(PORT), "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"],
    cwd=ROOT,
)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=2)
            break
        except Exception:
            time.sleep(1)
    print("server up", flush=True)

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        for sample, ask, name in TESTS:
            if (OUT / name).exists():
                print(name, "already exists, skipping", flush=True)
                continue
            for attempt in (1, 2):
                page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1)
                q = urllib.parse.urlencode({"sample": sample, "ask": ask})
                t = time.time()
                try:
                    page.goto(f"{URL}/?{q}")
                    # wait until the assistant reply is rendered (question + answer = 2 chat messages)
                    page.wait_for_function(
                        "document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 2",
                        timeout=300_000,
                    )
                    page.wait_for_timeout(1500)
                    reply = page.locator('[data-testid="stChatMessage"]').nth(1).inner_text()
                    model = page.locator("text=Model:").first.inner_text()
                    page.evaluate("document.querySelectorAll('*').forEach(e => { if (e.scrollTop) e.scrollTop = 0; })")
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(OUT / name), full_page=True)
                    results.append({"file": name, "paper": sample, "ask": ask, "model": model,
                                    "seconds": round(time.time() - t, 1), "reply": reply})
                    print(name, "done in", round(time.time() - t, 1), "s |", model, flush=True)
                    page.close()
                    time.sleep(20)  # stay under the free-tier embedding rate limit
                    break
                except Exception as e:
                    print(name, f"attempt {attempt} failed after", round(time.time() - t, 1), "s:", str(e)[:150], flush=True)
                    page.close()
        browser.close()
    out = ROOT / "screenshots" / "results.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    out.write_text(json.dumps(old + results, indent=2, ensure_ascii=False), encoding="utf-8")
finally:
    server.terminate()
    print("ALL DONE", flush=True)
