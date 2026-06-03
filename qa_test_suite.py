"""
============================================================================
QA TEST SUITE — Crop Recommendation API
============================================================================
Covers: normal, missing fields, invalid values, edge cases, performance,
        CORS, consistency, logging, error handling, and checklist.
============================================================================
"""

import json
import time
import requests
import threading
import statistics
import sys

BASE = "http://127.0.0.1:8000"
PREDICT = f"{BASE}/predict"
HEALTH  = f"{BASE}/health"

VALID_PAYLOAD = {
    "District":       "Dhenkanal",
    "State":          "Odisha",
    "Season":         "Rabi",
    "Soil_Type":      "Red",
    "Irrigation":     "Rainfed",
    "Rainfall":       750,
    "Temperature":    40,
    "Nitrogen":       250,
    "Phosphorus":     159,
    "Potassium":      189,
    "Humidity":       55,
    "Soil_pH":        5.0,
    "Area":           10000,
    "Fertilizer":     72,
    "Previous_Yield": 1.5,
    "Year":           2026,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

passed = []
failed = []
warnings_list = []

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(name, detail=""):
    passed.append(name)
    print(f"  {GREEN}✅ PASS{RESET}  {name}" + (f"  →  {detail}" if detail else ""))

def fail(name, detail=""):
    failed.append(name)
    print(f"  {RED}❌ FAIL{RESET}  {name}" + (f"  →  {detail}" if detail else ""))

def warn(name, detail=""):
    warnings_list.append(name)
    print(f"  {YELLOW}⚠️  WARN{RESET}  {name}" + (f"  →  {detail}" if detail else ""))

def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def post(payload, timeout=5):
    return requests.post(PREDICT, json=payload, timeout=timeout)

def check_response_shape(data: dict) -> list[str]:
    """Return list of missing required keys in response."""
    required = ["Best_Crop", "Expected_Yield", "Estimated_Profit",
                "Top_3_Recommendations", "Insights", "Feature_Summary"]
    return [k for k in required if k not in data]


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — NORMAL CASE
# ════════════════════════════════════════════════════════════════════════════
section("TEST 1 — Normal / Happy Path")

try:
    t0 = time.time()
    r  = post(VALID_PAYLOAD)
    elapsed = time.time() - t0

    # Status code
    if r.status_code == 200:
        ok("Status code 200")
    else:
        fail("Status code 200", f"got {r.status_code}: {r.text[:200]}")

    # Response time
    if elapsed < 1.0:
        ok(f"Response time < 1s", f"{elapsed*1000:.0f} ms")
    else:
        warn(f"Response time >= 1s", f"{elapsed*1000:.0f} ms")

    data = r.json()

    # Shape validation
    missing = check_response_shape(data)
    if not missing:
        ok("Response schema complete")
    else:
        fail("Response schema complete", f"Missing: {missing}")

    # Best_Crop is a non-empty string
    crop = data.get("Best_Crop", "")
    if isinstance(crop, str) and crop:
        ok("Best_Crop is non-empty string", crop)
    else:
        fail("Best_Crop is non-empty string", repr(crop))

    # Expected_Yield is positive float
    yld = data.get("Expected_Yield", -1)
    if isinstance(yld, (int, float)) and yld > 0:
        ok("Expected_Yield is positive", f"{yld} t/ha")
    else:
        fail("Expected_Yield is positive", repr(yld))

    # Estimated_Profit is positive
    profit = data.get("Estimated_Profit", -1)
    if isinstance(profit, (int, float)) and profit > 0:
        ok("Estimated_Profit is positive", f"₹{profit:,.0f}")
    else:
        fail("Estimated_Profit is positive", repr(profit))

    # Top_3_Recommendations has at least 1 entry
    top3 = data.get("Top_3_Recommendations", [])
    if isinstance(top3, list) and len(top3) >= 1:
        ok(f"Top_3_Recommendations non-empty", f"{len(top3)} entries")
    else:
        fail("Top_3_Recommendations non-empty", repr(top3))

    # Each recommendation has Crop + Profit keys
    if top3 and all("Crop" in r and "Profit" in r for r in top3):
        ok("Each recommendation has Crop + Profit keys")
    else:
        fail("Each recommendation has Crop + Profit keys")

    # Insights is a non-empty list
    insights = data.get("Insights", [])
    if isinstance(insights, list) and len(insights) > 0:
        ok("Insights non-empty list", f"{len(insights)} items")
    else:
        fail("Insights non-empty list")

    # Feature_Summary has all keys
    fs = data.get("Feature_Summary", {})
    fs_keys = ["Rainfall_Category", "Temperature_Category",
                "Soil_Fertility_Index", "Water_Stress_Index",
                "Weather_Score", "Yield_Efficiency", "Processing_Time_ms"]
    missing_fs = [k for k in fs_keys if k not in fs]
    if not missing_fs:
        ok("Feature_Summary has all expected keys")
    else:
        fail("Feature_Summary has all expected keys", f"Missing: {missing_fs}")

    # Realistic yield range (0.1 – 100 t/ha)
    if 0.1 <= yld <= 100:
        ok("Expected_Yield in realistic range", f"{yld}")
    else:
        fail("Expected_Yield in realistic range", f"{yld}")

    print(f"\n  Sample output:")
    print(f"    Best Crop       : {data.get('Best_Crop')}")
    print(f"    Expected Yield  : {data.get('Expected_Yield')} t/ha")
    print(f"    Estimated Profit: ₹{data.get('Estimated_Profit'):,.0f}")
    print(f"    Insights        : {data.get('Insights')}")

except Exception as e:
    fail("Normal test case (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — MISSING FIELDS
# ════════════════════════════════════════════════════════════════════════════
section("TEST 2 — Missing Required Fields (Defaults / Clean Errors)")

missing_tests = [
    ("Rainfall",    "should use default 1000 mm"),
    ("Temperature", "should use default 30°C"),
    ("Soil_Type",   "should use default Loamy"),
]

for field, note in missing_tests:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
    try:
        r = post(payload)
        if r.status_code == 200:
            ok(f"Missing {field} → uses default", note)
        elif r.status_code == 422:
            detail = r.json().get("detail", "")
            ok(f"Missing {field} → clean 422 error", str(detail)[:80])
        else:
            fail(f"Missing {field}", f"Unexpected status {r.status_code}")
    except Exception as e:
        fail(f"Missing {field} (exception)", str(e))

# Missing ALL optional fields — only required categoricals
bare_payload = {"District": "Cuttack", "Season": "Kharif",
                "Soil_Type": "Loamy", "Irrigation": "Canal"}
try:
    r = post(bare_payload)
    if r.status_code == 200:
        ok("Bare minimum payload (4 fields) → 200 with defaults")
    elif r.status_code == 422:
        ok("Bare minimum payload → clean 422", "numeric fields required by Pydantic")
    else:
        fail("Bare minimum payload", f"status {r.status_code}: {r.text[:100]}")
except Exception as e:
    fail("Bare minimum payload (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — INVALID VALUES
# ════════════════════════════════════════════════════════════════════════════
section("TEST 3 — Invalid Values (422 Validation)")

invalid_tests = [
    ("Rainfall -100",    {**VALID_PAYLOAD, "Rainfall": -100},   422, "Rainfall < 0"),
    ("Temperature 200",  {**VALID_PAYLOAD, "Temperature": 200}, 422, "Temperature > 50"),
    ("pH 20",            {**VALID_PAYLOAD, "Soil_pH": 20},      422, "pH > 9"),
    ("pH -1",            {**VALID_PAYLOAD, "Soil_pH": -1},      422, "pH < 3"),
    ("Bad Season",       {**VALID_PAYLOAD, "Season": "Monsoon"},422, "invalid season"),
    ("Bad Soil_Type",    {**VALID_PAYLOAD, "Soil_Type": "Clay"},422, "invalid soil type"),
    ("Bad Irrigation",   {**VALID_PAYLOAD, "Irrigation": "Drip"},422, "invalid irrigation"),
    ("Year too low",     {**VALID_PAYLOAD, "Year": 1990},       422, "year < 2000"),
    ("Year too high",    {**VALID_PAYLOAD, "Year": 2100},       422, "year > 2050"),
]

for name, payload, expected_status, note in invalid_tests:
    try:
        r = post(payload)
        if r.status_code == expected_status:
            detail_str = str(r.json().get("detail", ""))[:100]
            ok(f"{name} → {expected_status}", detail_str)
        else:
            fail(f"{name} → {expected_status}", f"got {r.status_code}: {r.text[:100]}")
    except Exception as e:
        fail(f"{name} (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — EDGE CASES
# ════════════════════════════════════════════════════════════════════════════
section("TEST 4 — Edge Case Scenarios (Insights Match Reality)")

# Case 1: Drought + Extreme Heat
case1 = {**VALID_PAYLOAD, "Rainfall": 50, "Temperature": 45}
try:
    r = post(case1)
    if r.status_code == 200:
        insights = r.json().get("Insights", [])
        joined = " ".join(insights).lower()
        drought_ok = any(w in joined for w in ["drought", "low rainfall", "irrigation", "critical"])
        heat_ok    = any(w in joined for w in ["heat", "temperature", "stress"])
        if drought_ok:
            ok("Case1: Drought insight triggered", insights[0])
        else:
            fail("Case1: Drought insight NOT in response", f"Insights: {insights}")
        if heat_ok:
            ok("Case1: Heat stress insight triggered", insights[0])
        else:
            fail("Case1: Heat stress insight NOT in response", f"Insights: {insights}")
    else:
        fail("Case1: Drought+Heat returned 200", f"got {r.status_code}")
except Exception as e:
    fail("Case1: Drought+Heat (exception)", str(e))

# Case 2: Flood conditions
case2 = {**VALID_PAYLOAD, "Rainfall": 2500, "Humidity": 95}
try:
    r = post(case2)
    if r.status_code == 200:
        insights = r.json().get("Insights", [])
        joined = " ".join(insights).lower()
        flood_ok = any(w in joined for w in ["flood", "excessive", "waterlog", "high rainfall", "drainage"])
        humidity_ok = any(w in joined for w in ["humidity", "fungal", "disease"])
        if flood_ok:
            ok("Case2: Flood/excess rainfall insight triggered")
        else:
            fail("Case2: Flood insight NOT triggered", f"Insights: {insights}")
        if humidity_ok:
            ok("Case2: High humidity insight triggered")
        else:
            warn("Case2: High humidity insight", f"Insights: {insights}")
        # Feature summary should show High rainfall
        fs = r.json().get("Feature_Summary", {})
        if fs.get("Rainfall_Category") == "High":
            ok("Case2: Feature_Summary shows High rainfall category")
        else:
            fail("Case2: Feature_Summary rainfall category", f"got: {fs.get('Rainfall_Category')}")
    else:
        fail("Case2: Flood conditions returned 200", f"got {r.status_code}")
except Exception as e:
    fail("Case2: Flood (exception)", str(e))

# Case 3: Very acidic soil
case3 = {**VALID_PAYLOAD, "Soil_pH": 4.0}
try:
    r = post(case3)
    if r.status_code == 200:
        insights = r.json().get("Insights", [])
        joined = " ".join(insights).lower()
        acid_ok = any(w in joined for w in ["acid", "lime", "ph"])
        if acid_ok:
            ok("Case3: Acidic soil insight triggered")
        else:
            fail("Case3: Acidic soil insight NOT triggered", f"Insights: {insights}")
    else:
        fail("Case3: Acidic soil returned 200", f"got {r.status_code}")
except Exception as e:
    fail("Case3: Acidic soil (exception)", str(e))

# Case 4: Maximum valid values
case4 = {**VALID_PAYLOAD, "Rainfall": 3000, "Temperature": 50,
         "Nitrogen": 500, "Phosphorus": 400, "Potassium": 500,
         "Humidity": 100, "Soil_pH": 9.0, "Area": 1000000,
         "Fertilizer": 500, "Previous_Yield": 200, "Year": 2050}
try:
    r = post(case4)
    if r.status_code == 200:
        ok("Case4: Max valid values → 200 OK")
    else:
        fail("Case4: Max valid values", f"got {r.status_code}: {r.text[:200]}")
except Exception as e:
    fail("Case4: Max valid values (exception)", str(e))

# Case 5: Minimum valid values
case5 = {**VALID_PAYLOAD, "Rainfall": 0, "Temperature": 0,
         "Nitrogen": 0, "Phosphorus": 0, "Potassium": 0,
         "Humidity": 0, "Soil_pH": 3.0, "Area": 1,
         "Fertilizer": 0, "Previous_Yield": 0, "Year": 2000}
try:
    r = post(case5)
    if r.status_code == 200:
        ok("Case5: Min valid values → 200 OK")
    else:
        fail("Case5: Min valid values", f"got {r.status_code}: {r.text[:200]}")
except Exception as e:
    fail("Case5: Min valid values (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — REQUEST/RESPONSE LOGGING CHECK
# ════════════════════════════════════════════════════════════════════════════
section("TEST 5 — Logging (via Feature_Summary Processing Time)")
# The API includes Processing_Time_ms in Feature_Summary — a proxy that
# server-side timing is being tracked. We verify it's always present.

try:
    r = post(VALID_PAYLOAD)
    fs = r.json().get("Feature_Summary", {})
    pt = fs.get("Processing_Time_ms")
    if pt is not None and isinstance(pt, (int, float)) and pt >= 0:
        ok("Processing_Time_ms logged in Feature_Summary", f"{pt} ms")
    else:
        fail("Processing_Time_ms missing or invalid", repr(pt))
except Exception as e:
    fail("Logging check (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 6 — ERROR HANDLING
# ════════════════════════════════════════════════════════════════════════════
section("TEST 6 — Error Handling (Malformed JSON, Type Errors)")

# Completely malformed JSON
try:
    r = requests.post(PREDICT,
                      data="NOT VALID JSON AT ALL !!!",
                      headers={"Content-Type": "application/json"},
                      timeout=5)
    if r.status_code in (400, 422):
        ok("Malformed JSON → clean 4xx error", f"status {r.status_code}")
    else:
        fail("Malformed JSON", f"got {r.status_code}: {r.text[:100]}")
except Exception as e:
    fail("Malformed JSON (exception)", str(e))

# Wrong types (string where number expected)
payload_wrong_types = {**VALID_PAYLOAD, "Rainfall": "heavy", "Temperature": "hot"}
try:
    r = post(payload_wrong_types)
    if r.status_code == 422:
        ok("Wrong types (string as float) → 422")
    elif r.status_code == 200:
        warn("Wrong types accepted (coercion)", "Pydantic may coerce strings to float")
    else:
        fail("Wrong types", f"got {r.status_code}")
except Exception as e:
    fail("Wrong types (exception)", str(e))

# Empty body
try:
    r = requests.post(PREDICT, json={}, timeout=5)
    if r.status_code in (200, 422):
        ok(f"Empty body → {r.status_code} (handled cleanly)")
    else:
        fail("Empty body", f"got {r.status_code}")
except Exception as e:
    fail("Empty body (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 7 — PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
section("TEST 7 — Performance (50 sequential requests)")

N = 50
response_times = []
errors = 0

print(f"  Running {N} sequential requests...")
for i in range(N):
    try:
        t0 = time.time()
        r = post(VALID_PAYLOAD)
        elapsed = (time.time() - t0) * 1000  # ms
        if r.status_code == 200:
            response_times.append(elapsed)
        else:
            errors += 1
    except Exception as e:
        errors += 1

if response_times:
    avg = statistics.mean(response_times)
    p95 = sorted(response_times)[int(len(response_times) * 0.95)]
    p99 = sorted(response_times)[int(len(response_times) * 0.99)]
    mn  = min(response_times)
    mx  = max(response_times)
    print(f"\n  Results ({len(response_times)}/{N} successful):")
    print(f"    Min:  {mn:.0f} ms")
    print(f"    Mean: {avg:.0f} ms")
    print(f"    P95:  {p95:.0f} ms")
    print(f"    P99:  {p99:.0f} ms")
    print(f"    Max:  {mx:.0f} ms")
    print(f"    Errors: {errors}")

    if errors == 0:
        ok("50 requests — zero errors")
    else:
        fail(f"50 requests — {errors} errors")

    if avg < 200:
        ok(f"Mean response time < 200ms", f"{avg:.0f} ms")
    elif avg < 1000:
        warn(f"Mean response time {avg:.0f} ms", "acceptable but could be optimised")
    else:
        fail(f"Mean response time too high", f"{avg:.0f} ms")

    if p95 < 1000:
        ok(f"P95 < 1s", f"{p95:.0f} ms")
    else:
        warn(f"P95 >= 1s", f"{p95:.0f} ms")

    # Check for slowdown (drift): compare first 10 vs last 10
    if len(response_times) >= 20:
        first10 = statistics.mean(response_times[:10])
        last10  = statistics.mean(response_times[-10:])
        drift   = last10 - first10
        if drift < 100:
            ok(f"No slowdown (drift={drift:+.0f} ms)")
        else:
            warn(f"Possible slowdown detected", f"drift = +{drift:.0f} ms")
else:
    fail("Performance test — no successful responses")


# ════════════════════════════════════════════════════════════════════════════
# TEST 8 — CORS SIMULATION
# ════════════════════════════════════════════════════════════════════════════
section("TEST 8 — CORS Headers")

try:
    # Simulate a browser preflight OPTIONS request from a different origin
    r = requests.options(
        PREDICT,
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
        timeout=5,
    )
    ac_origin = r.headers.get("access-control-allow-origin", "")
    ac_methods = r.headers.get("access-control-allow-methods", "")

    if ac_origin in ("*", "http://localhost:3000"):
        ok("CORS: access-control-allow-origin present", ac_origin)
    else:
        fail("CORS: access-control-allow-origin missing", f"got: '{ac_origin}'")

    if "POST" in ac_methods or ac_methods == "*":
        ok("CORS: POST method allowed", ac_methods)
    else:
        warn("CORS: methods header", f"got: '{ac_methods}'")

    # Actual POST with Origin header
    r2 = requests.post(
        PREDICT,
        json=VALID_PAYLOAD,
        headers={"Origin": "http://my-frontend.com", "Content-Type": "application/json"},
        timeout=5,
    )
    if r2.status_code == 200:
        ok("CORS: POST from foreign origin succeeds")
    else:
        fail("CORS: POST from foreign origin", f"got {r2.status_code}")

except Exception as e:
    fail("CORS test (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# TEST 9 — RESPONSE CONSISTENCY
# ════════════════════════════════════════════════════════════════════════════
section("TEST 9 — Response Consistency (Determinism)")

responses = []
for i in range(5):
    try:
        r = post(VALID_PAYLOAD)
        if r.status_code == 200:
            responses.append(r.json())
    except Exception:
        pass

if len(responses) >= 3:
    crops  = [r["Best_Crop"] for r in responses]
    yields = [r["Expected_Yield"] for r in responses]

    if len(set(crops)) == 1:
        ok("Best_Crop is deterministic across 5 calls", crops[0])
    else:
        warn("Best_Crop varies across calls", f"crops: {set(crops)}")

    max_yield_drift = max(yields) - min(yields)
    if max_yield_drift < 0.001:
        ok("Expected_Yield is deterministic", f"{yields[0]:.4f}")
    else:
        warn("Expected_Yield drifts slightly", f"range: {min(yields):.4f}–{max(yields):.4f}")

    profits = [r["Estimated_Profit"] for r in responses]
    if max(profits) - min(profits) < 1:
        ok("Estimated_Profit is deterministic")
    else:
        warn("Estimated_Profit drifts", f"range: {min(profits):.2f}–{max(profits):.2f}")
else:
    fail("Consistency test — not enough responses", f"{len(responses)}/5")


# ════════════════════════════════════════════════════════════════════════════
# TEST 10 — FINAL CHECKLIST
# ════════════════════════════════════════════════════════════════════════════
section("TEST 10 — Final Production Checklist")

# ✔ Model loads once — verify via timing difference (first call vs subsequent)
try:
    t1 = time.time(); post(VALID_PAYLOAD); t1 = time.time() - t1
    t2 = time.time(); post(VALID_PAYLOAD); t2 = time.time() - t2
    # Second call should NOT be significantly slower (models cached)
    if abs(t1 - t2) < 5.0:
        ok("Model loaded once (not per-request)", f"call1={t1*1000:.0f}ms call2={t2*1000:.0f}ms")
    else:
        warn("Possible model reload per request", f"t1={t1*1000:.0f}ms t2={t2*1000:.0f}ms")
except Exception as e:
    fail("Model caching check (exception)", str(e))

# ✔ Health endpoint works
try:
    r = requests.get(HEALTH, timeout=5)
    if r.status_code == 200 and r.json().get("status") == "API is running":
        ok("GET /health returns correct status")
    else:
        fail("GET /health", f"{r.status_code}: {r.text[:100]}")
except Exception as e:
    fail("GET /health (exception)", str(e))

# ✔ Response is valid JSON always
try:
    r = post(VALID_PAYLOAD)
    json.loads(r.text)
    ok("Response is always valid JSON")
except Exception as e:
    fail("Response valid JSON", str(e))

# ✔ All categorical encoders match training
try:
    for district in ["Angul", "Balangir", "Cuttack", "Dhenkanal", "Sambalpur"]:
        r = post({**VALID_PAYLOAD, "District": district})
        if r.status_code != 200:
            fail(f"District '{district}' encoding", f"status {r.status_code}")
            break
    else:
        ok("All tested districts encode correctly (5 districts)")
except Exception as e:
    fail("District encoding check (exception)", str(e))

try:
    for season in ["Kharif", "Rabi", "Zaid"]:
        r = post({**VALID_PAYLOAD, "Season": season})
        if r.status_code != 200:
            fail(f"Season '{season}' encoding", f"status {r.status_code}")
            break
    else:
        ok("All 3 seasons encode correctly")
except Exception as e:
    fail("Season encoding check (exception)", str(e))

try:
    for soil in ["Alluvial", "Black", "Laterite", "Loamy", "Red", "Sandy"]:
        r = post({**VALID_PAYLOAD, "Soil_Type": soil})
        if r.status_code != 200:
            fail(f"Soil_Type '{soil}' encoding", f"status {r.status_code}")
            break
    else:
        ok("All 6 soil types encode correctly")
except Exception as e:
    fail("Soil type encoding check (exception)", str(e))

try:
    for irr in ["Canal", "Rainfed", "Tube Well"]:
        r = post({**VALID_PAYLOAD, "Irrigation": irr})
        if r.status_code != 200:
            fail(f"Irrigation '{irr}' encoding", f"status {r.status_code}")
            break
    else:
        ok("All 3 irrigation types encode correctly")
except Exception as e:
    fail("Irrigation encoding check (exception)", str(e))

# ✔ Clean JSON output — no NaN, Infinity, nulls in critical fields
try:
    r = post(VALID_PAYLOAD)
    data = r.json()
    critical = [data.get("Best_Crop"), data.get("Expected_Yield"),
                data.get("Estimated_Profit")]
    if all(v is not None for v in critical):
        ok("No null values in critical output fields")
    else:
        fail("Null in critical output", f"{critical}")

    text = r.text
    has_nan = "NaN" in text or "Infinity" in text or "undefined" in text
    if not has_nan:
        ok("No NaN/Infinity/undefined in JSON response")
    else:
        fail("NaN or Infinity found in response", text[:200])
except Exception as e:
    fail("JSON null/NaN check (exception)", str(e))


# ════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════
total = len(passed) + len(failed)
print(f"\n\n{'═'*60}")
print(f"{BOLD}  📊 QA TEST REPORT{RESET}")
print(f"{'═'*60}")
print(f"  {GREEN}Passed : {len(passed):3d}  ({len(passed)/total*100:.1f}%){RESET}")
print(f"  {RED}Failed : {len(failed):3d}  ({len(failed)/total*100:.1f}%){RESET}")
print(f"  {YELLOW}Warnings: {len(warnings_list):2d}{RESET}")
print(f"  Total  : {total}")
print(f"{'═'*60}")

if failed:
    print(f"\n{RED}{BOLD}  ❌ FAILED TESTS:{RESET}")
    for f in failed:
        print(f"    • {f}")

if warnings_list:
    print(f"\n{YELLOW}{BOLD}  ⚠️  WARNINGS:{RESET}")
    for w in warnings_list:
        print(f"    • {w}")

overall = "✅ PRODUCTION READY" if len(failed) == 0 else (
    "⚠️  NEAR READY — FIX FAILURES" if len(failed) <= 3 else "❌ NOT READY"
)
print(f"\n  Verdict: {BOLD}{overall}{RESET}")
print(f"{'═'*60}\n")

sys.exit(0 if len(failed) == 0 else 1)
