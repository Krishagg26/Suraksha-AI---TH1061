# app.py
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# OSRM public demo server
OSRM_BASE = "http://router.project-osrm.org"  # demo; for production use your own routing service or paid API

@app.route("/")
def index():
    return render_template("index.html")

def osrm_route(coords):
    """
    coords: list of (lon,lat) tuples
    returns: dict JSON from OSRM route service
    """
    coord_str = ";".join([f"{c[0]},{c[1]}" for c in coords])
    url = f"{OSRM_BASE}/route/v1/driving/{coord_str}"
    params = {
        "overview": "full",
        "geometries": "polyline",
        "steps": "false",
        "annotations": "duration,distance"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@app.route("/route", methods=["POST"])
def route():
    data = request.get_json(force=True)
    amb = data.get("ambulance")
    patient = data.get("patient")
    hospitals = data.get("hospitals", [])
    choose_best = data.get("choose_best_hospital", True)

    if not amb or not patient or not hospitals:
        return jsonify({"error": "ambulance, patient and at least one hospital required"}), 400

    # convert to (lon,lat) pairs
    amb_ll = (amb[1], amb[0])
    pat_ll = (patient[1], patient[0])

    # Leg 1: ambulance -> patient
    try:
        res1 = osrm_route([amb_ll, pat_ll])
        r1 = res1["routes"][0]
        leg1 = {
            "distance_m": r1["distance"],
            "duration_s": r1["duration"],
            "geometry": r1["geometry"]
        }
    except Exception as e:
        return jsonify({"error": f"OSRM leg1 failure: {e}"}), 500

    # Leg 2: patient -> hospital(s)
    patient_to_hospital_results = []
    for idx, h in enumerate(hospitals):
        h_ll = (h[1], h[0])
        try:
            res = osrm_route([pat_ll, h_ll])
            r = res["routes"][0]
            info = {
                "index": idx,
                "distance_m": r["distance"],
                "duration_s": r["duration"],
                "geometry": r["geometry"]
            }
            patient_to_hospital_results.append(info)
        except Exception as e:
            patient_to_hospital_results.append({"index": idx, "error": str(e)})

    valid_results = [p for p in patient_to_hospital_results if "duration_s" in p]
    if not valid_results:
        return jsonify({"error": "All hospital route requests failed"}), 500

    if choose_best:
        best = min(valid_results, key=lambda x: x["duration_s"])
        best_idx = best["index"]
        leg2 = best
    else:
        leg2 = valid_results[0]
        best_idx = leg2["index"]

    total_distance = leg1["distance_m"] + leg2["distance_m"]
    total_duration = leg1["duration_s"] + leg2["duration_s"]

    out = {
        "leg1": leg1,
        "leg2": leg2,
        "total": {"distance_m": total_distance, "duration_s": total_duration},
        "selected_hospital_index": best_idx,
        "all_hospital_results": patient_to_hospital_results
    }
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
