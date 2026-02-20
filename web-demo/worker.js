import init, { run_main_pod_points } from "../pkg-web/pod2.js";

let initialized = false;
let initPromise = null;

async function ensureInit() {
  if (initialized) return;
  if (!initPromise) {
    initPromise = init()
      .then(() => {
        initialized = true;
        self.postMessage({ type: "ready" });
      })
      .catch((err) => {
        initPromise = null;
        throw err;
      });
  }
  await initPromise;
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg?.type !== "run") return;

  const mode = msg.mode === "real" ? "real" : "mock";
  const mock = mode === "mock";

  try {
    self.postMessage({ type: "status", text: "Loading wasm..." });
    await ensureInit();
    self.postMessage({ type: "status", text: `Proving (${mode})...` });

    const start = performance.now();
    const result = run_main_pod_points(mock);
    const elapsedMs = Math.round(performance.now() - start);

    self.postMessage({
      type: "result",
      mode,
      result,
      elapsedMs,
    });
  } catch (err) {
    const rendered =
      err instanceof Error
        ? `${err.name}: ${err.message}\n${err.stack ?? ""}`
        : String(err);
    self.postMessage({
      type: "error",
      mode,
      error: rendered,
    });
  }
};
