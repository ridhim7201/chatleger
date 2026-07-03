/* sw.js — ChatLedger service worker
   Caches the app shell (HTML + any static assets) on first load so the
   UI opens offline.  API calls (/upload, /results, /reset) are always
   sent to the network — cached data in SQLite is the offline store for
   extracted results, not the service worker cache.
*/

const CACHE = "chatledger-v1";

// Files that make up the app shell
const SHELL = ["/", "/manifest.json"];

// ── Install: pre-cache the shell ─────────────────────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL))
  );
  self.skipWaiting();
});

// ── Activate: delete old caches ───────────────────────────────────────────
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch: network-first for API, cache-first for shell ───────────────────
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Always hit the network for API calls
  if (url.pathname.startsWith("/upload") ||
      url.pathname.startsWith("/results") ||
      url.pathname.startsWith("/reset") ||
      url.pathname.startsWith("/health")) {
    return; // browser handles normally
  }

  // App shell: serve from cache, fall back to network
  event.respondWith(
    caches.match(event.request).then(
      (cached) => cached || fetch(event.request)
    )
  );
});