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

# QueueServer discovers this conventional global when it opens the worker.
RE = RunEngine({})


def _record(plan_name: str, **parameters) -> dict[str, str]:
    """Persist one simulated QueueServer request for inspection and confirmation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    request = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_name": plan_name,
        "parameters": parameters,
    }
    stem = f"{request['timestamp'].replace(':', '-')}_{request['id']}"
    json_path = OUTPUT_DIR / f"{stem}.json"
    text_path = OUTPUT_DIR / f"{stem}.txt"
    formatted_request = json.dumps(request, indent=2, default=str)
    json_path.write_text(formatted_request + "\n")
    text_path.write_text(
        "HXN QueueServer simulator receipt\n"
        f"Sent (UTC): {request['timestamp']}\n"
        f"Plan: {plan_name}\n"
        f"Request ID: {request['id']}\n\n"
        f"Input received:\n{formatted_request}\n"
    )
    print(f"[HXN SIM] {plan_name}: recorded request at {text_path}")
    return {
        "request_id": request["id"],
        "timestamp": request["timestamp"],
        "json_receipt": str(json_path.resolve()),
        "text_receipt": str(text_path.resolve()),
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
    yield from bps.sleep(0.01)
    return {"success": True, **receipt}


def piezos_to_zero():
    """Simulate resetting piezo stages."""
    receipt = _record("piezos_to_zero")
    yield from bps.sleep(0.01)
    return {"success": True, **receipt}


def move_relative(motor, distance):
    """Simulate an HXN relative motor move."""
    receipt = _record("move_relative", motor=motor, distance=distance)
    yield from bps.sleep(0.01)
    return {"success": True, **receipt}


def mov(*args):
    """Simulate Bluesky's multi-motor absolute move helper."""
    receipt = _record("mov", arguments=args)
    yield from bps.sleep(0.01)
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
    yield from bps.sleep(0.01)
    return {"success": True, **receipt}
