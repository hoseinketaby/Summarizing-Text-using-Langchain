const btn = document.getElementById("summarizeBtn");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

function showStatus(message, type) {
  statusBox.textContent = message;
  statusBox.className = `status ${type}`;
  statusBox.classList.remove("hidden");
}

function hideStatus() {
  statusBox.classList.add("hidden");
}

function showResult(text) {
  resultBox.textContent = text;
  resultBox.classList.remove("hidden");
}

function hideResult() {
  resultBox.classList.add("hidden");
}

btn.addEventListener("click", async () => {
  const apiKey = document.getElementById("apiKey").value;
  const url = document.getElementById("url").value;
  const language = document.getElementById("language").value;
  const model = document.getElementById("model").value;

  hideResult();

  // Client-side validation, mirroring the Streamlit checks
  if (!apiKey.trim() || !url.trim()) {
    showStatus("Please provide the important information", "error");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Waiting...";
  showStatus("Waiting...", "info");

  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        url: url,
        language: language,
        model: model,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      showStatus(data.error || "Something went wrong.", "error");
    } else {
      hideStatus();
      showResult(data.summary);
    }
  } catch (err) {
    showStatus(`Exception: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Summarize the content from YT or Website";
  }
});
