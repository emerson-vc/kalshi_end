// Combo Edge client: theme, SSE → HTMX triggers, toasts, live quote ages, persisted filters.
(function () {
  const root = document.documentElement;
  try {
    const t = localStorage.getItem("ce-theme");
    if (t === "dark" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches)) root.classList.add("dark");
  } catch (e) {}
  window.ceToggleTheme = function () {
    root.classList.toggle("dark");
    try { localStorage.setItem("ce-theme", root.classList.contains("dark") ? "dark" : "light"); } catch (e) {}
  };

  // Server-sent events: each event re-fetches the partials that listen for it.
  function connect() {
    const es = new EventSource("/events");
    ["header", "pipeline", "results", "run_started", "run_finished"].forEach(function (ev) {
      es.addEventListener(ev, function () {
        document.body.dispatchEvent(new CustomEvent("sse-" + ev));
        if (ev === "run_finished" || ev === "run_started") document.body.dispatchEvent(new CustomEvent("sse-results"));
      });
    });
    es.addEventListener("toast", function (e) {
      try { const d = JSON.parse(e.data); window.ceToast(d.message, d.level); } catch (x) {}
    });
    es.onerror = function () { es.close(); setTimeout(connect, 3000); };
  }
  document.addEventListener("DOMContentLoaded", connect);

  // Toasts (in-page only; no system notifications).
  window.ceToast = function (message, level) {
    const box = document.getElementById("toasts");
    if (!box) return;
    const el = document.createElement("div");
    const color = { success: "var(--pos)", error: "var(--neg)", warn: "var(--warn)" }[level] || "var(--accent)";
    el.className = "panel px-4 py-3 text-sm shadow-lg max-w-sm";
    el.style.borderLeft = "4px solid " + color;
    el.textContent = message;
    box.appendChild(el);
    setTimeout(function () { el.style.transition = "opacity .4s"; el.style.opacity = "0"; }, 5000);
    setTimeout(function () { el.remove(); }, 5600);
  };

  // Quote ages tick live and turn stale after 3 s.
  setInterval(function () {
    const now = Date.now() / 1000;
    document.querySelectorAll("[data-quote-ts]").forEach(function (el) {
      const ts = parseFloat(el.dataset.quoteTs);
      if (!ts) return;
      const age = now - ts;
      el.textContent = age < 120 ? age.toFixed(0) + "s" : age < 7200 ? Math.floor(age / 60) + "m" : Math.floor(age / 3600) + "h";
      el.classList.toggle("bad", age > 3);
      el.title = age > 3 ? "Stale: refresh before buying" : "Fresh";
    });
  }, 1000);

  // Copy legs (to find the combo in the Kalshi app).
  window.ceCopy = function (text, btn) {
    navigator.clipboard.writeText(text).then(function () {
      const old = btn.textContent; btn.textContent = "Copied"; setTimeout(function () { btn.textContent = old; }, 1200);
    });
  };

  // Found-table filters, remembered per browser.
  window.ceFilters = function (defaults) {
    let saved = {};
    // v2: the payout filter used to default to the 8x–50x band and hid anything paying more (e.g. a 78x
    // near-lock); old saved filters are dropped so the wide default applies.
    try { saved = JSON.parse(localStorage.getItem("ce-filters-v2") || "{}"); } catch (e) {}
    return Object.assign({
      lo: defaults.lo, hi: defaults.hi, family: "", game: "", minEdge: -100, window: "all",
      save() { try { localStorage.setItem("ce-filters-v2", JSON.stringify({ lo: this.lo, hi: this.hi, family: this.family, game: this.game, minEdge: this.minEdge, window: this.window })); } catch (e) {} },
      show(el) {
        const d = el.dataset;
        const pay = parseFloat(d.payout || "0");
        if (pay && (pay < this.lo || (this.hi < 1000 && pay > this.hi))) return false;  // top of the slider = no upper limit
        if (this.family && d.family !== this.family) return false;
        if (this.game && d.game !== this.game) return false;
        const edge = parseFloat(d.edge || "-999");
        if (edge < this.minEdge) return false;
        const hrs = (parseFloat(d.kickoff || "0") - Date.now() / 1000) / 3600;
        if (this.window === "24h" && hrs > 24) return false;
        if (this.window === "72h" && hrs > 72) return false;
        return true;
      },
    }, saved);
  };
})();
