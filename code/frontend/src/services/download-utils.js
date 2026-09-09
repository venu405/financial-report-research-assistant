const DEFAULT_DOWNLOAD_NAME = "financial-research-working-paper.md";

export function sanitizeDownloadFileName(fileName, fallback = DEFAULT_DOWNLOAD_NAME) {
  const clean = String(fileName || "")
    .replace(/[\u0000-\u001f<>:"/\\|?*]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .replace(/[. ]+$/, "");
  if (clean) return clean;
  return String(fallback || DEFAULT_DOWNLOAD_NAME)
    .replace(/[\u0000-\u001f<>:"/\\|?*]/g, "_")
    .replace(/^\.+/, "")
    .replace(/[. ]+$/, "") || DEFAULT_DOWNLOAD_NAME;
}

export function filenameFromContentDisposition(header, fallback = DEFAULT_DOWNLOAD_NAME) {
  const value = String(header || "");
  const encoded = value.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  const plain = value.match(/filename\s*=\s*(?:"([^"]+)"|([^;]+))/i);
  let candidate = (plain?.[1] || plain?.[2] || "").trim();
  if (encoded?.[1]) {
    try {
      candidate = decodeURIComponent(encoded[1].trim());
    } catch {
      candidate = encoded[1].trim();
    }
  }
  return sanitizeDownloadFileName(candidate, fallback);
}

export function downloadBlob(blob, fileName, options = {}) {
  const documentRef = options.documentRef || (typeof document === "undefined" ? null : document);
  const urlApi = options.urlApi || (typeof URL === "undefined" ? null : URL);
  if (!documentRef || !urlApi) throw new Error("当前环境不支持文件下载");

  const safeName = sanitizeDownloadFileName(fileName);
  const objectUrl = urlApi.createObjectURL(blob);
  const anchor = documentRef.createElement("a");
  anchor.href = objectUrl;
  anchor.download = safeName;
  anchor.style.display = "none";
  try {
    documentRef.body.appendChild(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    urlApi.revokeObjectURL(objectUrl);
  }
  return safeName;
}
