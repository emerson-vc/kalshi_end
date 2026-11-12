"""Start the dashboard: localhost only by default; ``--lan`` serves your home network behind the
passcode in COMBO_EDGE_LAN_PASSCODE."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser


def serve(port: int = 8000, lan: bool = False, open_browser: bool = True) -> int:
    import uvicorn

    from combo_edge.web.app import create_app

    if lan and not os.environ.get("COMBO_EDGE_LAN_PASSCODE"):
        print("--lan needs COMBO_EDGE_LAN_PASSCODE set in .env (the passcode your phone will enter).")
        return 2
    host = "0.0.0.0" if lan else "127.0.0.1"
    app = create_app(lan=lan, argv=sys.argv[1:])
    url = f"http://localhost:{port}"
    if open_browser and not os.environ.get("COMBO_EDGE_NO_BROWSER"):
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    print(f"Combo Edge dashboard on {url}" + (" (and your home network, passcode required)" if lan else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
