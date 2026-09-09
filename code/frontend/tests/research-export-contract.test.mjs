import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  downloadBlob,
  filenameFromContentDisposition,
  sanitizeDownloadFileName,
} from "../src/services/download-utils.js";

const companyApiSource = await readFile(new URL("../src/services/financial-metrics-api.ts", import.meta.url), "utf8");
const peerApiSource = await readFile(new URL("../src/services/peer-comparison-api.ts", import.meta.url), "utf8");
const companyViewSource = await readFile(new URL("../src/CompanyAnalysisView.vue", import.meta.url), "utf8");
const peerViewSource = await readFile(new URL("../src/ResearchPlaceholder.vue", import.meta.url), "utf8");

test("export APIs keep fixed paths, repeated query parameters, and token headers", () => {
  assert.match(companyApiSource, /\/kb\/company-analysis\/export/);
  assert.match(companyApiSource, /companyAnalysisQuery\(params\)/);
  assert.match(companyApiSource, /Content-Disposition/);
  assert.match(companyApiSource, /headers\(\)/);
  assert.match(peerApiSource, /\/kb\/peer-comparison\/export/);
  assert.match(peerApiSource, /peerComparisonQuery\(params\)/);
  assert.match(peerApiSource, /query\.append\("company_names"/);
  assert.match(peerApiSource, /query\.append\("metric_codes"/);
  assert.match(peerApiSource, /Content-Disposition/);
  assert.match(peerApiSource, /headers\(\)/);
});

test("export buttons use successful-result parameter snapshots", () => {
  assert.match(companyViewSource, /analysisExportParams\.value = \{ companyName, reportPeriod, comparisonPeriod \}/);
  assert.match(companyViewSource, /exportCompanyAnalysis\(analysisExportParams\.value\)/);
  assert.match(companyViewSource, /analysisExportLoading/);
  assert.match(companyViewSource, /导出研究底稿/);
  assert.match(peerViewSource, /peerExportParams\.value = \{ companyNames: \[\.\.\.companyNames\], reportPeriod, metricCodes: \[\.\.\.metricCodes\] \}/);
  assert.match(peerViewSource, /requestPeerComparisonExport\(peerExportParams\.value\)/);
  assert.match(peerViewSource, /peerExportLoading/);
  assert.match(peerViewSource, /导出研究底稿/);
});

test("download filenames are safe and prefer Content-Disposition", () => {
  assert.equal(
    filenameFromContentDisposition("attachment; filename*=UTF-8''%E7%A0%94%E7%A9%B6%E5%BA%95%E7%A8%BF.md", "fallback.md"),
    "研究底稿.md",
  );
  assert.equal(sanitizeDownloadFileName("../研究/底稿?.md"), "_研究_底稿_.md");
  assert.equal(filenameFromContentDisposition("attachment", "fallback.md"), "fallback.md");
});

test("downloadBlob uses a temporary object URL, download attribute, and revoke", () => {
  const calls = [];
  const revoked = [];
  const anchor = {
    style: {},
    click() { calls.push("click"); },
    remove() { calls.push("remove"); },
  };
  const documentRef = {
    createElement(tag) {
      assert.equal(tag, "a");
      return anchor;
    },
    body: { appendChild(element) { calls.push(["append", element]); } },
  };
  const urlApi = {
    createObjectURL(blob) {
      assert.ok(blob instanceof Blob);
      calls.push("create");
      return "blob:test";
    },
    revokeObjectURL(url) { revoked.push(url); },
  };
  const fileName = downloadBlob(new Blob(["# 研究底稿"]), "研究/底稿.md", { documentRef, urlApi });
  assert.equal(fileName, "研究_底稿.md");
  assert.equal(anchor.href, "blob:test");
  assert.equal(anchor.download, "研究_底稿.md");
  assert.deepEqual(revoked, ["blob:test"]);
  assert.deepEqual(calls.map((item) => Array.isArray(item) ? item[0] : item), ["create", "append", "click", "remove"]);
});
