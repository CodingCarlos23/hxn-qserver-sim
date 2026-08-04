# HXN QueueServer Simulator

This is a standalone, hardware-free QueueServer environment for testing how
X-AutoMap-HXN communicates with BlueSky QueueServer. It never connects to HXN
motors, detectors, Tiled, databroker, or beamline file systems.

## Quick start

Clone the repository, then run:

```bash
git clone https://github.com/<your-github-username>/hxn-qserver-sim.git
cd hxn-qserver-sim
pixi install
pixi run generate-lists
pixi run manager
```

If you already have a local copy, replace the first two commands with `cd` to
the directory where you cloned it.

Leave this terminal open while testing. The simulator listens locally at
`tcp://localhost:60615`, which is the default address used by X-AutoMap-HXN.

## Start it

Redis must be installed and running locally. Start it with your operating
system's service manager; for example, on many Linux systems:

```bash
sudo systemctl start redis-server
```

```bash
cd hxn-qserver-sim
pixi install
pixi run generate-lists
pixi run manager
```

In a second terminal, open the simulated worker and inspect its status:

```bash
cd hxn-qserver-sim
pixi run environment-open
pixi run status
```

The QueueServer control endpoint is local (`tcp://localhost:60615` by default),
which is what `REManagerAPI()` in X-AutoMap-HXN targets by default.

## What is simulated

The startup file registers the HXN-specific plan names used by X-AutoMap-HXN:

- `fly2d_qserver_scan_export`
- `piezos_to_zero`
- `move_relative`
- `mov`
- `recover_zp_csan_pos`

Each accepted plan creates both a JSON receipt and a human-readable `.txt`
receipt in `output/`, timestamped in UTC. The plan result also contains the
receipt paths, so a future GUI submission workflow can retrieve confirmation
from QueueServer history rather than relying only on files. The function
signatures deliberately match the client calls in X-AutoMap-HXN, including the 19 arguments of
`fly2d_qserver_scan_export`.

## Send one test plan

With the manager running and the worker opened:

```bash
cd hxn-qserver-sim
pixi run python -c "from bluesky_queueserver_api import BPlan; from bluesky_queueserver_api.zmq import REManagerAPI; rm = REManagerAPI(); rm.item_add(BPlan('fly2d_qserver_scan_export', 'laptop-test', ['fs'], 'zpssx', -1, 1, 10, 'zpssy', -1, 1, 10, 0.01)); rm.queue_start()"
```

Then inspect the JSON file in `output/`. Stop the manager with `Ctrl-C` in its
terminal. The queue uses the isolated Redis prefix `hxn-qserver-sim`, so it
does not share queue records with other local QueueServer instances.

## Scope

This validates the client-to-QueueServer interaction: plan serialization,
QueueServer validation, queuing, starting, execution, and status handling. It
does not validate HXN hardware behavior, real plan logic, data export, or
beamline authorization.

## TODO

- Generate three fake element-like XRF TIFF images with known blobs, then use
  them to test the complete local workflow:

  ```text
  fake TIFFs → X-AutoMap detection → boxes → Queue Selected Plans
  → local QueueServer simulator → dated receipts
  ```

  This should validate OpenCV processing, blob detection, union-box generation,
  and fine-scan planning without beamline data.

- Move QueueServer submission and status polling to a background GUI task so
  opening a worker or starting a queue never pauses the interface.
- Display QueueServer completion/failure status and the simulator receipt path
  directly in X-AutoMap.
- Add a visible, configurable QueueServer endpoint in X-AutoMap instead of
  relying only on the default `tcp://localhost:60615`.
