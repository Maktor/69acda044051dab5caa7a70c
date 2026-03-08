import xml.etree.ElementTree as ET
from collections import defaultdict

XML_FILE = "ei_bsrt_m_r2__custom_20423128_sdmx_generic_2_1.xml"

tree = ET.parse(XML_FILE)
root = tree.getroot()

ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
ns = {
    "g":   f"{ns_uri}",
    "msg": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "gen": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
}

print("=" * 70)
print("EUROSTAT RETAIL TRADE SURVEY — OLS REGRESSION")
print("Dataset : EI_BSRT_M_R2")
print("Filter  : geo = EU27_2020  |  s_adj = NSA")
print("X (predictor) : BS-RAS  (assessment of current inventory levels)")
print("Y (target)    : BS-RPE  (selling price expectations)")
print("=" * 70)

data = {}

for series in root.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Series"):
    keys = {}
    for key_val in series.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Value"):
        keys[key_val.attrib.get("id")] = key_val.attrib.get("value")

    geo   = keys.get("geo",   keys.get("GEO",   ""))
    s_adj = keys.get("s_adj", keys.get("S_ADJ", ""))
    indic = keys.get("indic", keys.get("INDIC", ""))

    if geo != "EU27_2020" or s_adj != "NSA":
        continue
    if indic not in ("BS-RPE", "BS-RAS"):
        continue

    for obs in series.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Obs"):
        dim = obs.find("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsDimension")
        val = obs.find("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsValue")
        if dim is None or val is None:
            continue
        period = dim.attrib.get("value")
        try:
            v = float(val.attrib.get("value"))
        except (TypeError, ValueError):
            continue
        data[(period, indic)] = v

periods_ras = {p for (p, i) in data if i == "BS-RAS"}
periods_rpe = {p for (p, i) in data if i == "BS-RPE"}
common = sorted(periods_ras & periods_rpe)

X = [data[(p, "BS-RAS")] for p in common]
Y = [data[(p, "BS-RPE")] for p in common]
n = len(common)

print(f"\nMatched observations (common periods): {n}")
print(f"\n{'Period':<12}  {'BS-RAS (X)':>14}  {'BS-RPE (Y)':>14}")
print("-" * 44)
for p, x, y in zip(common, X, Y):
    print(f"{p:<12}  {x:>14.10f}  {y:>14.10f}")

print("\n" + "=" * 70)
print("STEP-BY-STEP OLS CALCULATIONS (10 decimal places)")
print("=" * 70)

mean_x = sum(X) / n
mean_y = sum(Y) / n
print(f"\nStep 1 — Compute means")
print(f"  n        = {n}")
print(f"  mean(X)  = sum(X) / n = {sum(X):.10f} / {n} = {mean_x:.10f}")
print(f"  mean(Y)  = sum(Y) / n = {sum(Y):.10f} / {n} = {mean_y:.10f}")

dx = [xi - mean_x for xi in X]
dy = [yi - mean_y for yi in Y]

SS_xx  = sum(dxi**2        for dxi      in dx)
SS_yy  = sum(dyi**2        for dyi      in dy)
SS_xy  = sum(dxi * dyi     for dxi, dyi in zip(dx, dy))

print(f"\nStep 2 — Deviations from mean (first 5 shown)")
print(f"  {'Period':<12}  {'dx = X - mean_x':>20}  {'dy = Y - mean_y':>20}")
print("  " + "-" * 56)
for p, dxi, dyi in list(zip(common, dx, dy))[:5]:
    print(f"  {p:<12}  {dxi:>20.10f}  {dyi:>20.10f}")
print(f"  ... ({n - 5} more rows)")

print(f"\nStep 3 — Sums of squares / cross-products")
print(f"  SS_xx = Σ(dx²)    = {SS_xx:.10f}")
print(f"  SS_yy = Σ(dy²)    = {SS_yy:.10f}")
print(f"  SS_xy = Σ(dx·dy)  = {SS_xy:.10f}")

slope     = SS_xy / SS_xx
intercept = mean_y - slope * mean_x

print(f"\nStep 4 — OLS coefficients")
print(f"  slope (β₁)     = SS_xy / SS_xx")
print(f"                 = {SS_xy:.10f} / {SS_xx:.10f}")
print(f"                 = {slope:.10f}")
print(f"  intercept (β₀) = mean_y - slope × mean_x")
print(f"                 = {mean_y:.10f} - {slope:.10f} × {mean_x:.10f}")
print(f"                 = {intercept:.10f}")

SS_res = sum((yi - (slope * xi + intercept))**2 for xi, yi in zip(X, Y))
R2     = 1 - SS_res / SS_yy

print(f"\nStep 5 — Goodness of fit")
print(f"  SS_res (residual SS) = {SS_res:.10f}")
print(f"  R²                   = 1 - SS_res/SS_yy")
print(f"                       = 1 - {SS_res:.10f} / {SS_yy:.10f}")
print(f"                       = {R2:.10f}")

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(f"  OLS slope coefficient (β₁) = {slope:.10f}")
print(f"  Rounded to 2 decimal places = {round(slope, 2)}")
print("=" * 70)
