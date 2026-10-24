/**
 * VERITAS — research console.
 *
 * Everything that comes from the API (answers, claims, snippets, titles, URLs) is rendered
 * through DOM nodes and textContent, never innerHTML.
 */
(() => {
  "use strict";

  // Grounding badge thresholds, as documented in the README.
  const GROUNDED_MIN = 0.85;
  const PARTIAL_MIN = 0.7;
  const MIN_QUESTION_LENGTH = 3;

  const VERDICTS = {
    supported: { label: "Supported", className: "supported" },
    partially_supported: { label: "Partially supported", className: "partial" },
    unsupported: { label: "Unsupported", className: "unsupported" },
  };
  const MODE_LABELS = { nli: "Claude NLI", "llm-as-judge": "Claude as judge", "local-nli": "the local NLI model" };

  // ── Pure helpers ─────────────────────────────────────────

  function truncate(text, max) {
    const value = String(text ?? "");
    return value.length > max ? `${value.slice(0, max).trimEnd()}…` : value;
  }

  /** Absolute http(s) URLs only, so a javascript: link can never be rendered. */
  function safeHttpUrl(url) {
    try {
      const parsed = new URL(String(url));
      return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
    } catch {
      return null;
    }
  }

  /** The verifier copies claims without citation markers or final punctuation; try those forms too. */
  function claimVariants(claimText) {
    const exact = String(claimText ?? "").trim();
    const withoutCitations = exact.replace(/\s*(\[\d+\])+/g, "").trim();
    const withoutPunctuation = withoutCitations.replace(/[.!?;:,]+$/, "").trim();
    return [...new Set([exact, withoutCitations, withoutPunctuation])].filter((v) => v.length >= 4);
  }

  /** Non-overlapping [start, end) spans of the answer where each claim appears, in order. */
  function findClaimRanges(answer, claims) {
    const ranges = [];
    const overlaps = (start, end) => ranges.some((r) => start < r.end && end > r.start);
    // Longest first, so a short claim can't take the middle of a longer one.
    const ordered = [...claims].sort((a, b) => String(b.claim_text ?? "").length - String(a.claim_text ?? "").length);
    for (const claim of ordered) {
      let found = null;
      for (const variant of claimVariants(claim.claim_text)) {
        for (let start = answer.indexOf(variant); start !== -1 && !found; start = answer.indexOf(variant, start + 1)) {
          const end = start + variant.length;
          if (!overlaps(start, end)) found = { start, end, claim };
        }
        if (found) break;
      }
      if (found) ranges.push(found);
    }
    return ranges.sort((a, b) => a.start - b.start);
  }

  /** Split text into plain strings and {citation: n} tokens for each [n] marker. */
  function tokenizeCitations(text) {
    const tokens = [];
    let last = 0;
    for (const match of text.matchAll(/\[(\d+)\]/g)) {
      if (match.index > last) tokens.push(text.slice(last, match.index));
      tokens.push({ citation: Number(match[1]) });
      last = match.index + match[0].length;
    }
    if (last < text.length) tokens.push(text.slice(last));
    return tokens;
  }

  function badgeFor(grounding) {
    if (grounding.status !== "verified" || typeof grounding.score !== "number") {
      const label = grounding.status === "no_context" ? "No evidence found" : "Verification unavailable";
      return { variant: "unverified", label, value: "—" };
    }
    const value = `${Math.round(grounding.score * 100)}%`;
    if (grounding.score >= GROUNDED_MIN) return { variant: "fully-grounded", label: "Grounded", value };
    if (grounding.score >= PARTIAL_MIN) return { variant: "partially-grounded", label: "Partly grounded", value };
    return { variant: "warning", label: "Needs review", value };
  }

  // ── Browser UI ───────────────────────────────────────────

  function init() {
    const $ = (id) => document.getElementById(id);
    const state = { history: [], inFlight: false };

    const headerStatus = $("header-status");
    const statusDot = $("status-dot");
    const statusText = $("status-text");
    const docSelect = $("doc-select");
    const docHelp = $("doc-help");
    const refreshDocsBtn = $("refresh-docs-btn");
    const verifyGrounding = $("verify-grounding");
    const verificationMode = $("verification-mode");
    const questionInput = $("question-input");
    const includeWeb = $("include-web");
    const includeArxiv = $("include-arxiv");
    const submitBtn = $("submit-btn");
    const btnLabel = $("btn-label");
    const resultCard = $("result-card");
    const errorBanner = $("error-banner");
    const resultPlaceholder = $("result-placeholder");
    const resultContent = $("result-content");
    const badgeContainer = $("grounding-badge-container");
    const answerDisplay = $("answer-display");
    const auditDetails = $("audit-logs-details");
    const auditContent = $("audit-logs-content");
    const retrievalList = $("retrieval-list");
    const sourcesList = $("sources-list");
    const exportActions = $("export-actions");
    const exportCsvBtn = $("export-csv-btn");
    const exportPdfBtn = $("export-pdf-btn");
    const exportError = $("export-error");
    const historyEmpty = $("history-empty");
    const historyTableWrapper = $("history-table-wrapper");
    const historyTableBody = $("history-table-body");
    const runtimeInfo = $("runtime-info");
    const uploadRow = $("upload-row");
    const uploadInput = $("upload-input");
    const uploadStatus = $("upload-status");

    function el(tag, props = {}, children = []) {
      const node = document.createElement(tag);
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null) continue;
        if (key === "text") node.textContent = value;
        else if (key === "className") node.className = value;
        else node.setAttribute(key, value);
      }
      node.append(...children);
      return node;
    }

    function setVisible(node, visible) {
      node.classList.toggle("hidden", !visible);
    }

    function showMessage(node, message) {
      node.textContent = message;
      setVisible(node, Boolean(message));
    }

    /** FastAPI errors carry {"detail": "..."} or, for validation errors, a list of issues. */
    async function errorMessage(response) {
      try {
        const body = await response.json();
        if (typeof body.detail === "string") return body.detail;
        if (Array.isArray(body.detail)) return body.detail.map((issue) => issue.msg).join("; ");
      } catch {
        // not JSON
      }
      return `Request failed (HTTP ${response.status})`;
    }

    // ── Status & documents ──

    function setStatus(variant, text, detail) {
      statusDot.className = `status-dot ${variant}`;
      statusText.textContent = text;
      headerStatus.title = detail || "";
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = await response.json();
        if (body.ready) setStatus("online", "Ready", "");
        else setStatus("degraded", "Degraded", body.detail || "The research service is not ready");
      } catch {
        setStatus("error", "Offline", "The API is unreachable");
      }
    }

    function renderDocOptions(documents) {
      const selected = docSelect.value;
      docSelect.replaceChildren(
        el("option", { value: "", text: "All documents" }),
        ...documents.map((name) => el("option", { value: name, text: name })),
      );
      if (documents.includes(selected)) docSelect.value = selected;
      docHelp.textContent = documents.length
        ? `${documents.length} indexed document${documents.length === 1 ? "" : "s"}`
        : "No documents are indexed yet";
    }

    async function fetchDocuments() {
      refreshDocsBtn.disabled = true;
      try {
        const response = await fetch("/v1/documents");
        if (!response.ok) throw new Error(await errorMessage(response));
        const body = await response.json();
        renderDocOptions(body.documents || []);
      } catch (err) {
        docHelp.textContent = `Could not load the document list: ${err.message}`;
      } finally {
        refreshDocsBtn.disabled = false;
      }
    }

    // ── Backends & uploads ──

    /** Offer only the verification modes the server can run, and show which models are active. */
    async function loadConfig() {
      try {
        const response = await fetch("/v1/config");
        if (!response.ok) return;
        const config = await response.json();
        const modes = new Set(config.verification_modes || []);
        for (const option of verificationMode.options) {
          option.disabled = !modes.has(option.value);
          const why = option.value === "local-nli" ? " (not installed on the server)" : " (needs an API key)";
          if (option.disabled && !option.textContent.endsWith(why)) option.textContent += why;
        }
        if (modes.has(config.default_verification_mode)) verificationMode.value = config.default_verification_mode;
        const model = String(config.llm_model || "").split("/").pop();
        const search = config.vector_store === "pinecone" ? "Pinecone" : "BM25 + vector index";
        runtimeInfo.textContent = `${config.llm_provider === "local" ? "Local" : "Claude"}: ${model} · ${search}`;
        setVisible(uploadRow, Boolean(config.uploads_enabled));
      } catch {
        // the console still works with the defaults
      }
    }

    async function uploadDocument() {
      const file = uploadInput.files && uploadInput.files[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      uploadStatus.textContent = `Indexing ${file.name}…`;
      try {
        const response = await fetch("/v1/documents", { method: "POST", body: form });
        if (!response.ok) throw new Error(await errorMessage(response));
        const body = await response.json();
        uploadStatus.textContent = `Indexed ${body.document} (${body.chunks} chunk${body.chunks === 1 ? "" : "s"})`;
        await fetchDocuments();
        docSelect.value = body.document;
      } catch (err) {
        uploadStatus.textContent = `Upload failed: ${err.message}`;
      } finally {
        uploadInput.value = ""; // allow uploading the same file again
      }
    }

    // ── Result rendering ──

    function focusEvidence(index) {
      const card = document.getElementById(`evidence-${index}`);
      if (!card) return;
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      card.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
      card.classList.remove("flash");
      void card.offsetWidth; // restart the highlight animation
      card.classList.add("flash");
    }

    /** A [n] marker that jumps to its passage, the same in the answer and in the claims table. */
    function citeNode(index, citable) {
      if (!citable.has(index)) {
        // A citation with no matching passage is itself a grounding problem: make it visible.
        return el("span", { className: "cite cite-missing", title: "No evidence passage has this number", text: `[${index}]` });
      }
      const link = el("a", { className: "cite", href: `#evidence-${index}`, text: `[${index}]` });
      link.addEventListener("click", (event) => {
        event.preventDefault();
        focusEvidence(index);
      });
      return link;
    }

    function appendWithCitations(parent, text, citable) {
      for (const token of tokenizeCitations(text)) {
        parent.append(typeof token === "string" ? token : citeNode(token.citation, citable));
      }
    }

    function renderAnswer(answer, claims, citable) {
      answerDisplay.replaceChildren();
      const flagged = claims.filter((claim) => claim.verdict !== "supported");
      let cursor = 0;
      for (const { start, end, claim } of findClaimRanges(answer, flagged)) {
        appendWithCitations(answerDisplay, answer.slice(cursor, start), citable);
        const verdict = VERDICTS[claim.verdict] || VERDICTS.unsupported;
        const mark = el("mark", {
          className: `claim claim-${verdict.className}`,
          title: claim.reason ? `${verdict.label}: ${claim.reason}` : verdict.label,
        });
        appendWithCitations(mark, answer.slice(start, end), citable);
        answerDisplay.append(mark);
        cursor = end;
      }
      appendWithCitations(answerDisplay, answer.slice(cursor), citable);
    }

    function renderBadge(grounding) {
      badgeContainer.replaceChildren();
      if (!grounding) return;
      const { variant, label, value } = badgeFor(grounding);
      const claims = grounding.claims || [];
      const flagged = claims.filter((claim) => claim.verdict !== "supported").length;
      const summary =
        grounding.status === "verified"
          ? `${claims.length} claim${claims.length === 1 ? "" : "s"} checked with ${MODE_LABELS[grounding.mode] || "the verifier"}, ${flagged} flagged`
          : grounding.detail || "";
      badgeContainer.append(
        el("div", { className: `summary ${variant}` }, [
          el("span", { className: "tag", text: label }),
          el("span", { className: "summary-text", text: summary }),
          el("span", { className: "summary-score" }, ["Grounding score ", el("strong", { text: value })]),
        ]),
      );
    }

    function renderAudit(grounding, citable) {
      const claims = grounding?.claims || [];
      const rows = claims.map((claim) => {
        const verdict = VERDICTS[claim.verdict] || VERDICTS.unsupported;
        const refs = claim.citation_indices?.length ? claim.citation_indices.map((index) => citeNode(Number(index), citable)) : ["none"];
        return el("tr", {}, [
          el("td", {}, [el("span", { className: `tag ${verdict.className}`, text: verdict.label })]),
          el("td", { className: "claim-text", text: claim.claim_text }),
          el("td", { className: "claim-refs" }, refs),
          el("td", { className: "claim-reason", text: claim.reason || "" }),
        ]);
      });
      const header = el("tr", {}, ["Verdict", "Claim", "Cites", "Reason"].map((name) => el("th", { scope: "col", text: name })));
      auditContent.replaceChildren(
        el("table", { className: "table claims-table" }, [el("thead", {}, [header]), el("tbody", {}, rows)]),
      );
      setVisible(auditDetails, claims.length > 0);
    }

    function evidenceCard({ index, title, url, tag, meta, text }) {
      const href = url ? safeHttpUrl(url) : null;
      const titleNode = href
        ? el("a", { className: "evidence-title", href, target: "_blank", rel: "noopener noreferrer", text: title })
        : el("span", { className: "evidence-title", text: title });
      const header = el("div", { className: "evidence-header" }, [el("span", { className: "evidence-index", text: `[${index}]` }), titleNode]);
      if (tag) header.append(el("span", { className: "tag", text: tag }));
      if (meta) header.append(el("span", { className: "evidence-meta", text: meta }));
      return el("article", { className: "evidence-item", id: `evidence-${index}` }, [header, el("p", { className: "evidence-text", text })]);
    }

    function renderEvidence(hits, sources) {
      retrievalList.replaceChildren(
        ...(hits.length
          ? hits.map((hit) =>
              evidenceCard({
                index: hit.citation_index,
                title: hit.document,
                meta: `${hit.page_num ? `page ${hit.page_num} · ` : ""}score ${Number(hit.score || 0).toFixed(2)}`,
                text: hit.preview,
              }),
            )
          : [el("p", { className: "muted-note", text: "No passages in your documents matched this question" })]),
      );
      sourcesList.replaceChildren(
        ...(sources.length
          ? sources.map((source) =>
              evidenceCard({
                index: source.citation_index,
                title: source.title || "External source",
                url: source.url,
                tag: source.kind === "arxiv" ? "arXiv" : "Web",
                text: truncate(source.snippet, 300),
              }),
            )
          : [el("p", { className: "muted-note", text: "No web or arXiv sources were searched or found" })]),
      );
    }

    function renderResult(result) {
      setVisible(resultPlaceholder, false);
      setVisible(resultContent, true);
      // Number evidence the way the API does (chunks, then sources) if an older API omits it.
      const hits = (result.retrieval || []).map((hit, i) => ({ ...hit, citation_index: hit.citation_index ?? i + 1 }));
      const sources = (result.sources || []).map((source, i) => ({
        ...source,
        citation_index: source.citation_index ?? hits.length + i + 1,
      }));
      const citable = new Set([...hits, ...sources].map((item) => item.citation_index));
      renderBadge(result.grounding);
      renderAnswer(result.answer || "", result.grounding?.claims || [], citable);
      renderAudit(result.grounding, citable);
      renderEvidence(hits, sources);
    }

    // ── Research pipeline ──

    function setBusy(busy) {
      state.inFlight = busy;
      submitBtn.disabled = busy;
      submitBtn.classList.toggle("loading", busy);
      btnLabel.textContent = busy ? "Answering…" : "Ask";
      resultCard.setAttribute("aria-busy", String(busy));
    }

    function toSessionRecord(payload, result) {
      const grounding = result.grounding;
      const verified = grounding?.status === "verified" && typeof grounding.score === "number";
      return {
        document: payload.selected_document || "All documents",
        question: payload.question,
        answer: result.answer || "",
        grounding_score: verified ? grounding.score : null,
        flagged_claims: verified ? grounding.claims.filter((claim) => claim.verdict !== "supported") : [],
        created_at: result.created_at,
      };
    }

    async function runResearch() {
      if (state.inFlight) return; // Ctrl+Enter can fire while a request is running
      const question = questionInput.value.trim();
      if (question.length < MIN_QUESTION_LENGTH) {
        showMessage(errorBanner, `Enter a question of at least ${MIN_QUESTION_LENGTH} characters`);
        questionInput.focus();
        return;
      }

      const payload = {
        question,
        selected_document: docSelect.value || null,
        include_web: includeWeb.checked,
        include_arxiv: includeArxiv.checked,
        top_k: 5,
        verify_grounding: verifyGrounding.checked,
        verification_mode: verificationMode.value,
      };

      setBusy(true);
      showMessage(errorBanner, "");
      try {
        const response = await fetch("/v1/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const result = await response.json();
        renderResult(result);
        state.history.push(toSessionRecord(payload, result));
        renderHistory();
      } catch (err) {
        showMessage(errorBanner, `Couldn't get an answer: ${err.message}`);
      } finally {
        setBusy(false);
        checkHealth();
      }
    }

    // ── Session history & exports ──

    function renderHistory() {
      const empty = state.history.length === 0;
      setVisible(historyEmpty, empty);
      setVisible(historyTableWrapper, !empty);
      setVisible(exportActions, !empty);
      historyTableBody.replaceChildren(
        ...state.history.map((item, i) =>
          el("tr", {}, [
            el("td", { className: "num", text: String(i + 1) }),
            el("td", { text: item.document }),
            el("td", { text: item.question }),
            el("td", { className: "muted", text: truncate(item.answer, 100) }),
            el("td", { className: "num", text: item.grounding_score === null ? "—" : `${Math.round(item.grounding_score * 100)}%` }),
          ]),
        ),
      );
    }

    async function exportFile(endpoint, filename, button) {
      if (state.history.length === 0) return;
      button.disabled = true;
      showMessage(exportError, "");
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ records: state.history, metadata: { app: "veritas-web" } }),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const url = URL.createObjectURL(await response.blob());
        const link = el("a", { href: url, download: filename });
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000); // revoking synchronously can cancel the download
      } catch (err) {
        showMessage(exportError, `Export failed: ${err.message}`);
      } finally {
        button.disabled = false;
      }
    }

    // ── Wiring ──

    function syncVerificationMode() {
      verificationMode.disabled = !verifyGrounding.checked;
    }

    refreshDocsBtn.addEventListener("click", fetchDocuments);
    uploadInput.addEventListener("change", uploadDocument);
    submitBtn.addEventListener("click", runResearch);
    exportCsvBtn.addEventListener("click", () => exportFile("/v1/export/csv", "research_session.csv", exportCsvBtn));
    exportPdfBtn.addEventListener("click", () => exportFile("/v1/export/pdf", "research_session.pdf", exportPdfBtn));
    verifyGrounding.addEventListener("change", syncVerificationMode);
    questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        runResearch();
      }
    });

    // Show the shortcut the way this platform writes it; the handler above accepts either key.
    if (/mac|iphone|ipad/i.test(navigator.userAgentData?.platform || navigator.platform || "")) {
      const mod = $("shortcut-mod");
      mod.textContent = "⌘";
      mod.title = "Command";
    }

    syncVerificationMode(); // browsers may restore the checkbox state on reload
    loadConfig();
    checkHealth();
    fetchDocuments();
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", init);
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { badgeFor, claimVariants, findClaimRanges, safeHttpUrl, tokenizeCitations, truncate };
  }
})();
