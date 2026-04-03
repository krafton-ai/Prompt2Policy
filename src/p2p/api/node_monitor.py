"""Background polling of remote SSH node resources.

Provides cached snapshots used by the ``/resources/nodes`` endpoint.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from datetime import datetime, timezone

from p2p.api.schemas import NodeGpuInfo, NodeResourceSnapshot
from p2p.scheduler.ssh_utils import load_nodes, ssh_base_cmd

logger = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds

_cache: dict[str, NodeResourceSnapshot] = {}


def get_cached_snapshots() -> list[NodeResourceSnapshot]:
    """Return the most recently polled snapshots for all known nodes."""
    return list(_cache.values())


def _poll_node(node: dict) -> NodeResourceSnapshot:
    """SSH into *node* and collect CPU / memory / GPU stats."""
    node_id = node["node_id"]
    now = datetime.now(timezone.utc).isoformat()

    # Remote script that collects system stats and prints JSON.
    # Kept compact for SSH transport; runs as a single python3 -c call.
    script = "\n".join(
        [
            "import json,os,shutil",
            "c=os.cpu_count() or 0",
            "la=list(os.getloadavg())",
            "m={}",
            "try:",
            " with open('/proc/meminfo') as f:",
            "  d={k.strip():v.strip() for k,v in (l.split(':',1) for l in f if ':' in l)}",
            "  mt=int(d.get('MemTotal','0 kB').split()[0])//1024",
            "  ma=int(d.get('MemAvailable','0 kB').split()[0])//1024",
            "  m={'total':mt,'available':ma,'used':mt-ma}",
            "except Exception: m={'total':0,'available':0,'used':0}",
            "gpus=[]",
            "if shutil.which('nvidia-smi'):",
            " import subprocess as sp",
            " try:",
            "  q='index,name,utilization.gpu,memory.used,"
            "memory.total,temperature.gpu,power.draw,power.limit'",
            "  r=sp.run(['nvidia-smi','--query-gpu='+q,"
            "'--format=csv,noheader,nounits'],"
            "capture_output=True,text=True,timeout=5)",
            "  for l in r.stdout.strip().splitlines():",
            "   p=[x.strip() for x in l.split(',')]",
            "   gpus.append({'index':int(p[0]),'name':p[1],",
            "    'utilization':float(p[2]),",
            "    'memory_used_mb':float(p[3]),",
            "    'memory_total_mb':float(p[4]),",
            "    'temperature':float(p[5]),",
            "    'power_draw_w':float(p[6]),",
            "    'power_limit_w':float(p[7])})",
            " except Exception: pass",
            "print(json.dumps({'cpu_count':c,'load_avg':la,'mem':m,'gpus':gpus}))",
        ]
    )

    cmd = [*ssh_base_cmd(node), f"python3 -c {shlex.quote(script)}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return NodeResourceSnapshot(
                node_id=node_id,
                online=False,
                timestamp=now,
                error=result.stderr.strip()[:200],
            )
        data = json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
        return NodeResourceSnapshot(
            node_id=node_id,
            online=False,
            timestamp=now,
            error=str(e)[:200],
        )

    mem = data.get("mem", {})
    gpu_list = [NodeGpuInfo(**g) for g in data.get("gpus", [])]
    cpu_count = data.get("cpu_count", 0)
    load_avg = data.get("load_avg", [])
    cpu_pct = (load_avg[0] / cpu_count * 100) if cpu_count and load_avg else 0.0

    return NodeResourceSnapshot(
        node_id=node_id,
        online=True,
        timestamp=now,
        cpu_count=cpu_count,
        cpu_percent_avg=round(cpu_pct, 1),
        load_avg=load_avg,
        mem_total_mb=mem.get("total", 0),
        mem_used_mb=mem.get("used", 0),
        mem_available_mb=mem.get("available", 0),
        gpus=gpu_list,
    )


def poll_all_nodes() -> None:
    """Poll every enabled node and update the cache."""
    nodes = load_nodes()
    for node in nodes:
        if not node.get("enabled", True):
            continue
        snapshot = _poll_node(node)
        _cache[node["node_id"]] = snapshot
