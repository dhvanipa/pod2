const runMockBtn = document.getElementById("run-mock");
const runRealBtn = document.getElementById("run-real");
const resetBtn = document.getElementById("reset");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

const worker = new Worker("./worker.js", { type: "module" });
let running = false;

function setRunning(next) {
  running = next;
  runMockBtn.disabled = next;
  runRealBtn.disabled = next;
}

function appendLine(line) {
  outputEl.textContent += `${line}\n`;
}

worker.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "ready":
      statusEl.textContent = "Worker ready";
      statusEl.className = "ok";
      break;
    case "status":
      statusEl.textContent = msg.text;
      statusEl.className = "";
      break;
    case "result":
      setRunning(false);
      statusEl.textContent = `Done (${msg.elapsedMs} ms)`;
      statusEl.className = "ok";
      appendLine(`[${msg.mode}] ${msg.result}`);
      break;
    case "error":
      setRunning(false);
      statusEl.textContent = "Failed";
      statusEl.className = "err";
      appendLine(`[${msg.mode}] ERROR: ${msg.error}`);
      break;
    default:
      break;
  }
};

worker.onerror = (event) => {
  setRunning(false);
  statusEl.textContent = "Worker crashed";
  statusEl.className = "err";
  appendLine(`WORKER ERROR: ${event.message}`);
};

function run(mode) {
  if (running) return;
  setRunning(true);
  statusEl.textContent = `Running ${mode} prover in worker...`;
  statusEl.className = "";
  worker.postMessage({ type: "run", mode });
}

runMockBtn.addEventListener("click", () => run("mock"));
runRealBtn.addEventListener("click", () => run("real"));
resetBtn.addEventListener("click", () => {
  outputEl.textContent = "";
  statusEl.textContent = "Idle";
  statusEl.className = "";
});
