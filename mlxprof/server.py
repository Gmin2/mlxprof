"""Local viewer. Two lines in your script, then look in the browser.

This is a viewer for runs, not a monitor. It holds the last N runs in memory and
serves a page that polls them. Nothing is written to disk and nothing leaves the
machine.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mlxprof.auto import AutoProfiler
from mlxprof.core import load_state, machine_info

_profiler = None
_server = None

PAGE = """<!doctype html><meta charset=utf-8><title>mlxprof</title>
<style>
:root{--bg:#fbfaf8;--fg:#1a1a1a;--dim:#6b6b6b;--line:#e4e1dc;--card:#fff;--accent:#b4552d}
@media(prefers-color-scheme:dark){:root{--bg:#151514;--fg:#e8e6e3;--dim:#918d87;--line:#2c2b29;--card:#1d1d1b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:18px 22px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:15px;font-weight:600;letter-spacing:.02em}
.sub{color:var(--dim);margin-top:4px}
.warn{color:var(--accent);margin-top:6px}
main{padding:14px 22px;max-width:960px}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-weight:500;color:var(--dim);padding:6px 10px;border-bottom:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td{padding:7px 10px;border-bottom:1px solid var(--line)}
tr.run{cursor:pointer}
tr.run:hover td{background:var(--card)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.detail td{background:var(--card);padding:12px 16px}
.bar{display:inline-block;height:8px;background:var(--accent);opacity:.75;vertical-align:middle;border-radius:1px}
.kind{display:grid;grid-template-columns:180px 60px 1fr;gap:10px;align-items:center;padding:2px 0}
.empty{color:var(--dim);padding:30px 10px}
.mode{color:var(--dim);font-size:11px;margin-top:10px}
</style>
<header>
  <h1>mlxprof</h1>
  <div class=sub id=machine>...</div>
  <div class=warn id=warn></div>
</header>
<main>
<table><thead><tr>
  <th>run</th><th class=num>total</th><th class=num>calls</th>
  <th class=num>modules</th><th class=num>peak</th><th></th>
</tr></thead><tbody id=rows></tbody></table>
<div class=empty id=empty>waiting for a model to run ...</div>
</main>
<script>
let open_ = null;
function fmt(n,d=1){return n.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})}
async function tick(){
  const r = await fetch('/api/runs'); const d = await r.json();
  document.getElementById('machine').textContent = d.machine;
  document.getElementById('warn').textContent = d.warning || '';
  const rows = document.getElementById('rows');
  document.getElementById('empty').style.display = d.runs.length ? 'none' : 'block';
  rows.innerHTML = '';
  for (const run of d.runs.slice().reverse()) {
    const tr = document.createElement('tr'); tr.className = 'run';
    tr.innerHTML = `<td>#${run.id}</td><td class=num>${fmt(run.total_ms,2)} ms</td>`
      + `<td class=num>${run.calls}</td><td class=num>${run.modules}</td>`
      + `<td class=num>${fmt(run.peak_mb,0)} MB</td>`
      + `<td>${run.detail ? '' : 'run level'}</td>`;
    tr.onclick = () => { open_ = open_ === run.id ? null : run.id; tick(); };
    rows.appendChild(tr);
    if (open_ === run.id) {
      const max = Math.max(...run.kinds.map(k => run.detail ? k.ms : k.calls), 1);
      const body = run.kinds.map(k => {
        const v = run.detail ? k.ms : k.calls;
        const label = run.detail ? fmt(k.ms,2)+' ms' : k.calls+'x';
        return `<div class=kind><span>${k.name}</span><span class=num>${label}</span>`
             + `<span><i class=bar style="width:${v/max*100}%"></i></span></div>`;
      }).join('');
      const note = run.detail
        ? 'detail mode: eval forced at every boundary, so these inflate the run'
        : 'run level: no eval forced, so the total is honest and per layer times are not recorded';
      const tr2 = document.createElement('tr'); tr2.className = 'detail';
      tr2.innerHTML = `<td colspan=6>${body}<div class=mode>${note}</div></td>`;
      rows.appendChild(tr2);
    }
  }
}
tick(); setInterval(tick, 1000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/runs"):
            m = machine_info()
            topo = "+".join(f"{lv['cores']}{lv['name'][0]}" for lv in m["levels"])
            st = load_state()
            warn = ""
            if st and st["busy"]:
                warn = (f"load average {st['load1']:.1f} on {st['cores']} cores, "
                        "the machine is busy and these numbers are low")
            body = json.dumps(
                {
                    "machine": f"{m['brand']}  {topo}",
                    "warning": warn,
                    "runs": _profiler.runs if _profiler else [],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port=7878, detail=False, open_browser=True, keep=50):
    """Instrument everything and serve the runs at localhost:port.

    Call once, before you build or load a model. Returns the profiler so you can
    read .runs directly if you would rather not use the browser.
    """
    global _profiler, _server
    if _profiler is None:
        _profiler = AutoProfiler(detail=detail, keep=keep).install()
    if _server is None:
        _server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=_server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}"
        print(f"mlxprof serving at {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
    return _profiler
