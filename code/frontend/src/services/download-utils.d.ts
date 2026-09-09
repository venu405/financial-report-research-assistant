export function sanitizeDownloadFileName(fileName: unknown, fallback?: string): string;
export function filenameFromContentDisposition(header: unknown, fallback?: string): string;
export function downloadBlob(blob: Blob, fileName: unknown, options?: {
  documentRef?: Document;
  urlApi?: typeof URL;
}): string;
