// Unit tests for the web console's pure helpers. Run with: node --test tests/ui
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const { badgeFor, claimVariants, findClaimRanges, safeHttpUrl, tokenizeCitations, truncate } = require("../../ui/web/app.js");

const ANSWER = "Revenue grew 12% year-over-year [1]. The margin was 24% [2]. Costs fell.";

test("claims are located even though the verifier drops citation markers", () => {
  const claims = [
    { claim_text: "The margin was 24%", verdict: "unsupported" },
    { claim_text: "Revenue grew 12% year-over-year.", verdict: "partially_supported" },
  ];
  const ranges = findClaimRanges(ANSWER, claims);
  assert.deepEqual(
    ranges.map((r) => ANSWER.slice(r.start, r.end)),
    ["Revenue grew 12% year-over-year", "The margin was 24%"],
  );
});

test("overlapping claims do not produce nested highlights", () => {
  const claims = [
    { claim_text: "Revenue grew", verdict: "unsupported" },
    { claim_text: "Revenue grew 12% year-over-year", verdict: "unsupported" },
  ];
  const ranges = findClaimRanges(ANSWER, claims);
  assert.equal(ranges.length, 1);
  assert.equal(ANSWER.slice(ranges[0].start, ranges[0].end), "Revenue grew 12% year-over-year");
});

test("a repeated claim text takes its next free occurrence", () => {
  const text = "Sales rose. Sales rose.";
  const ranges = findClaimRanges(text, [
    { claim_text: "Sales rose.", verdict: "unsupported" },
    { claim_text: "Sales rose.", verdict: "unsupported" },
  ]);
  assert.deepEqual(ranges.map((r) => r.start), [0, 12]);
});

test("claims that are not in the answer are skipped", () => {
  assert.deepEqual(findClaimRanges(ANSWER, [{ claim_text: "Paraphrased beyond recognition", verdict: "unsupported" }]), []);
});

test("claim variants drop citations and final punctuation", () => {
  assert.deepEqual(claimVariants(" Costs fell [3][4]. "), ["Costs fell [3][4].", "Costs fell.", "Costs fell"]);
  assert.deepEqual(claimVariants("ok"), []);
});

test("citation markers are tokenized", () => {
  assert.deepEqual(tokenizeCitations("A [1][2] b [x] c [10]"), ["A ", { citation: 1 }, { citation: 2 }, " b [x] c ", { citation: 10 }]);
  assert.deepEqual(tokenizeCitations("no citations"), ["no citations"]);
});

test("only absolute http(s) links are allowed", () => {
  assert.equal(safeHttpUrl("https://arxiv.org/abs/2201.00978"), "https://arxiv.org/abs/2201.00978");
  assert.equal(safeHttpUrl("javascript:alert(1)"), null);
  assert.equal(safeHttpUrl("data:text/html,hi"), null);
  assert.equal(safeHttpUrl("/relative/path"), null);
  assert.equal(safeHttpUrl(undefined), null);
});

test("truncate adds an ellipsis only when it cuts", () => {
  assert.equal(truncate("short", 10), "short");
  assert.equal(truncate("a longer sentence", 8), "a longer…");
  assert.equal(truncate(null, 5), "");
});

test("badge thresholds match the documented 0.85 / 0.70 cut-offs", () => {
  const variant = (score) => badgeFor({ status: "verified", score }).variant;
  assert.equal(variant(1), "fully-grounded");
  assert.equal(variant(0.85), "fully-grounded");
  assert.equal(variant(0.84), "partially-grounded");
  assert.equal(variant(0.7), "partially-grounded");
  assert.equal(variant(0.69), "warning");
  assert.equal(badgeFor({ status: "verified", score: 0.5 }).value, "50%");
  assert.equal(badgeFor({ status: "no_context", score: null }).variant, "unverified");
  assert.equal(badgeFor({ status: "failed", score: null }).label, "Verification unavailable");
});
