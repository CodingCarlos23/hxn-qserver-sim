"""Safe, local stand-ins for the HXN QueueServer plans used by X-AutoMap-HXN.

This file is loaded by ``start-re-manager --startup-dir startup``.  It contains
no EPICS, Tiled, databroker, detector, or motor connections. Each plan records
the requested action to local JSON and text receipt files, then yields a short
Bluesky sleep plan.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bluesky import RunEngine
from bluesky import plan_stubs as bps


OUTPUT_DIR = Path(os.environ.get("HXN_QSERVER_SIM_OUTPUT", "output"))
LOG_FILE = OUTPUT_DIR / "log.txt"
RECORDS_FILE = OUTPUT_DIR / "records.jsonl"

# QueueServer discovers this conventional global when it opens the worker.
RE = RunEngine({})


def _scan_type(plan_name: str, parameters: dict) -> str:
    """Derive a human-readable scan type label from the plan name and parameters."""
    if plan_name == "fly2d_qserver_scan_export":
        roi = parameters.get("roi_json", "")
        if roi and roi not in ('""', ""):
            try:
                roi_dict = json.loads(roi)
                if any(v != 0.0 for v in roi_dict.values()):
                    return "FINE_SCAN"
            except Exception:
                pass
        return "COARSE_SCAN"
    if plan_name == "move_relative":
        return "TILE_MOVE"
    if plan_name == "mov":
        return "STAGE_RESET"
    if plan_name == "piezos_to_zero":
        return "PIEZO_RESET"
    if plan_name == "recover_zp_csan_pos":
        return "RECOVER_POS"
    return plan_name.upper()


def _log_line(plan_name: str, parameters: dict, timestamp: str) -> str:
    """Build a concise single log line for the plan."""
    ts = timestamp[:19].replace("T", " ")
    scan_type = _scan_type(plan_name, parameters)

    if plan_name == "fly2d_qserver_scan_export":
        label = parameters.get("label", "")
        m1, m1s, m1e, m1n = parameters.get("mot1",""), parameters.get("mot1_start",""), parameters.get("mot1_end",""), parameters.get("mot1_points","")
        m2, m2s, m2e, m2n = parameters.get("mot2",""), parameters.get("mot2_start",""), parameters.get("mot2_end",""), parameters.get("mot2_points","")
        roi = parameters.get("roi_json", "")
        roi_part = f"  roi={roi}" if roi and roi not in ('""', "") else ""
        detail = (
            f"label={label}  "
            f"{m1}[{m1s}→{m1e}, {m1n}pts]  "
            f"{m2}[{m2s}→{m2e}, {m2n}pts]"
            f"{roi_part}"
        )
    elif plan_name == "move_relative":
        detail = f"motor={parameters.get('motor','')}  distance={parameters.get('distance','')}"
    elif plan_name == "mov":
        args = parameters.get("arguments", ())
        detail = "  ".join(f"{args[i]}={args[i+1]}" for i in range(0, len(args)-1, 2))
    elif plan_name == "piezos_to_zero":
        detail = ""
    elif plan_name == "recover_zp_csan_pos":
        detail = f"ref_scan_id={parameters.get('ref_scan_id','')}  zp_move_flag={parameters.get('zp_move_flag','')}  smar_move_flag={parameters.get('smar_move_flag','')}"
    else:
        detail = "  ".join(f"{k}={v}" for k, v in parameters.items())

    return f"{ts}  {scan_type:<12}  {plan_name:<30}  {detail}\n"


def _record(plan_name: str, **parameters) -> dict[str, str]:
    """Persist one simulated QueueServer request to the shared log and records file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    request = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_name": plan_name,
        "parameters": parameters,
    }
    with open(LOG_FILE, "a") as lf:
        lf.write(_log_line(plan_name, parameters, request["timestamp"]))
    with open(RECORDS_FILE, "a") as rf:
        rf.write(json.dumps(request, default=str) + "\n")
    print(f"[HXN SIM] {plan_name}")
    return {
        "request_id": request["id"],
        "timestamp": request["timestamp"],
    }


def fly2d_qserver_scan_export(
    label,
    det_names,
    mot1,
    mot1_s,
    mot1_e,
    mot1_n,
    mot2,
    mot2_s,
    mot2_e,
    mot2_n,
    exp_t,
    roi_json="",
    scan_id="",
    zp_move_flag=0,
    smar_move_flag=0,
    ic1_count=6000,
    elem_list_json="[]",
    export_norm="sclr1_ch4",
    data_wd=".",
):
    """Simulate the HXN coarse/fine scan plan without controlling hardware."""
    receipt = _record(
        "fly2d_qserver_scan_export",
        label=label,
        det_names=det_names,
        mot1=mot1,
        mot1_start=mot1_s,
        mot1_end=mot1_e,
        mot1_points=mot1_n,
        mot2=mot2,
        mot2_start=mot2_s,
        mot2_end=mot2_e,
        mot2_points=mot2_n,
        exposure_seconds=exp_t,
        roi_json=roi_json,
        source_scan_id=scan_id,
        zp_move_flag=zp_move_flag,
        smar_move_flag=smar_move_flag,
        ic1_count=ic1_count,
        elem_list_json=elem_list_json,
        export_norm=export_norm,
        requested_data_directory=data_wd,
    )
    yield from bps.sleep(0.05)
    return {"success": True, **receipt}


def piezos_to_zero():
    """Simulate resetting piezo stages."""
    receipt = _record("piezos_to_zero")
    yield from bps.sleep(0.05)
    return {"success": True, **receipt}


def move_relative(motor, distance):
    """Simulate an HXN relative motor move."""
    receipt = _record("move_relative", motor=motor, distance=distance)
    yield from bps.sleep(0.05)
    return {"success": True, **receipt}


def mov(*args):
    """Simulate Bluesky's multi-motor absolute move helper."""
    receipt = _record("mov", arguments=args)
    yield from bps.sleep(0.05)
    return {"success": True, **receipt}


def recover_zp_csan_pos(ref_scan_id, zp_move_flag=0, smar_move_flag=1, move_base=1):
    """Simulate restoring an HXN zone-plate/sample position."""
    receipt = _record(
        "recover_zp_csan_pos",
        ref_scan_id=ref_scan_id,
        zp_move_flag=zp_move_flag,
        smar_move_flag=smar_move_flag,
        move_base=move_base,
    )
    yield from bps.sleep(0.05)
    return {"success": True, **receipt}
