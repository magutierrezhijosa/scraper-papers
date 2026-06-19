"""
WRI Research Papers Scraper
============================
Async scraper that extracts research publications from wri.org/research
using the sitemap as the URL source (the listing endpoint returns 403).
Saves results to CSV with all available metadata.

Usage:
    python scraper.py          # full run (with checkpoint resume)
    python scraper.py --fresh  # ignore existing checkpoint, start fresh
"""

import asyncio
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import aiohttp
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SITEMAP_URLS = [
    f"https://www.wri.org/sitemap.xml?page={i}" for i in range(1, 6)
]

RESEARCH_URL_PATTERN = re.compile(r"https://www\.wri\.org/research/.+")
EXCLUDE_PATTERNS = [
    re.compile(p)
    for p in [
        r"/research/excellence",
        r"/research/permissions-licensing",
        r"/research/permissions-request-form",
        r"/research/insights",
    ]
]

CONCURRENCY = 5        # parallel requests
CHECKPOINT_INTERVAL = 50
CHECKPOINT_FILE = "checkpoint.json"
OUTPUT_CSV = "wri_research_papers.csv"
ERROR_LOG = "wri_errors.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    url: str
    title: str = ""
    full_title: str = ""
    publication_date: str = ""
    pdf_url: str = ""
    authors: str = ""          # semicolon-separated
    institution: str = ""
    description: str = ""
    doi: str = ""
    og_title: str = ""
    og_description: str = ""
    wri_published: str = ""
    wri_authors: str = ""      # semicolon-separated
    topics: str = ""
    scrape_timestamp: str = ""
    status: str = "ok"         # ok | error | http_xxx


# ---------------------------------------------------------------------------
# Checkpoint (track by URL to support async out-of-order completion)
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    papers: list[Paper]
    errors: list[dict]
    done_urls: set[str]        # URLs that have been processed


def load_checkpoint() -> Checkpoint:
    if not os.path.exists(CHECKPOINT_FILE):
        return Checkpoint(papers=[], errors=[], done_urls=set())

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ckpt = Checkpoint(
        papers=[Paper(**p) for p in data.get("papers", [])],
        errors=data.get("errors", []),
        done_urls=set(data.get("done_urls", [])),
    )
    log.info(
        "Checkpoint loaded: %d papers, %d errors, %d done URLs",
        len(ckpt.papers), len(ckpt.errors), len(ckpt.done_urls),
    )
    return ckpt


def save_checkpoint(ckpt: Checkpoint) -> None:
    data = {
        "papers": [asdict(p) for p in ckpt.papers],
        "errors": ckpt.errors,
        "done_urls": list(ckpt.done_urls),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Checkpoint saved: %d papers, %d done URLs", len(ckpt.papers), len(ckpt.done_urls))


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "url", "title", "full_title", "publication_date", "pdf_url",
    "authors", "institution", "doi", "topics", "description",
    "og_title", "og_description", "wri_published", "wri_authors",
    "scrape_timestamp", "status",
]


def save_csv(papers: list[Paper]) -> None:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for p in papers:
            writer.writerow(asdict(p))
    log.info("CSV saved: %s (%d rows)", OUTPUT_CSV, len(papers))


def append_errors(errors: list[dict]) -> None:
    file_exists = os.path.exists(ERROR_LOG)
    with open(ERROR_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "error", "timestamp"])
        if not file_exists:
            writer.writeheader()
        for err in errors:
            writer.writerow(err)


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------


def fetch_sitemap_urls() -> list[str]:
    """Download all sitemap pages (sync, 5 quick requests) and return research URLs."""
    import requests as sync_requests

    all_urls: list[str] = []
    seen: set[str] = set()

    for sitemap_url in SITEMAP_URLS:
        try:
            log.info("Fetching sitemap: %s", sitemap_url)
            resp = sync_requests.get(sitemap_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.content, "xml")
            locs = soup.find_all("loc")

            for loc in locs:
                url = loc.get_text(strip=True)
                if not url:
                    continue
                if not RESEARCH_URL_PATTERN.match(url):
                    continue
                if any(p.search(url) for p in EXCLUDE_PATTERNS):
                    continue
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)

            log.info("  -> %d research URLs on this page (total: %d)", len(all_urls), len(all_urls))

        except Exception as e:
            log.error("Failed to fetch sitemap %s: %s", sitemap_url, e)

    log.info("Total unique research URLs found: %d", len(all_urls))
    return all_urls


# ---------------------------------------------------------------------------
# Individual page parsing (blocking — runs in thread pool)
# ---------------------------------------------------------------------------


def parse_publication(html: str, url: str) -> Paper:
    """Extract metadata from a single publication HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    paper = Paper(url=url)

    def meta(name: str) -> str | None:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content", "") if tag else None

    def meta_list(name: str) -> list[str]:
        return [t.get("content", "") for t in soup.find_all("meta", attrs={"name": name}) if t.get("content")]

    paper.title = meta("citation_title") or ""
    paper.publication_date = meta("citation_publication_date") or ""
    paper.pdf_url = meta("citation_pdf_url") or ""
    paper.doi = meta("citation_doi") or ""
    paper.institution = meta("citation_technical_report_institution") or ""
    paper.full_title = meta("wri_full_title") or ""
    paper.wri_published = meta("wri_published") or ""
    paper.description = meta("description") or ""

    authors = meta_list("citation_author")
    if authors:
        paper.authors = "; ".join(authors)

    wri_authors = meta_list("wri_author")
    if wri_authors:
        paper.wri_authors = "; ".join(wri_authors)

    # OG tags
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title:
        paper.og_title = og_title.get("content", "")
    og_desc = soup.find("meta", attrs={"property": "og:description"})
    if og_desc:
        paper.og_description = og_desc.get("content", "")

    # Topics from links
    topic_links = soup.select(
        "a[href*='/topics/'], a[href*='/climate/'], a[href*='/food/'], "
        "a[href*='/forest/'], a[href*='/water/'], a[href*='/cities/'], "
        "a[href*='/energy/'], a[href*='/ocean/']"
    )
    topic_set: set[str] = set()
    for link in topic_links:
        text = link.get_text(strip=True)
        if text and len(text) < 100:
            topic_set.add(text)
    if topic_set:
        paper.topics = "; ".join(sorted(topic_set))

    paper.scrape_timestamp = datetime.now(timezone.utc).isoformat()
    return paper


# ---------------------------------------------------------------------------
# Async fetch + parse (single URL)
# ---------------------------------------------------------------------------


async def fetch_one(
    sem: asyncio.Semaphore,
    url: str,
    session: aiohttp.ClientSession,
) -> Paper | dict:
    """Fetch and parse one research page. Returns Paper on success, error dict on failure."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                status = resp.status
                html = await resp.text()

            if status == 200:
                # BeautifulSoup is CPU-bound — run in thread pool
                paper = await asyncio.to_thread(parse_publication, html, url)
                paper.status = "ok"
                return paper
            else:
                return {"url": url, "error": f"HTTP {status}", "timestamp": datetime.now(timezone.utc).isoformat()}

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# ---------------------------------------------------------------------------
# Main orchestrator (async)
# ---------------------------------------------------------------------------


async def scrape_all_async(fresh: bool = False) -> None:
    """Orchestrate full async scrape."""
    # Ensure fresh state
    if fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Fresh start: deleted existing checkpoint")

    # Step 1: get all URLs from sitemap
    all_urls = fetch_sitemap_urls()
    if not all_urls:
        log.error("No URLs found in sitemap. Aborting.")
        return

    # Step 2: load checkpoint (if any)
    ckpt = load_checkpoint()
    pending = [u for u in all_urls if u not in ckpt.done_urls]
    total = len(all_urls)
    done_count = total - len(pending)

    log.info(
        "Starting scrape: %d total, %d already done, %d pending",
        total, done_count, len(pending),
    )

    if not pending:
        log.info("All URLs already scraped. Nothing to do.")
        _print_summary(ckpt.papers, total)
        return

    # Step 3: async scrape
    batch_errors: list[dict] = []
    sem = asyncio.Semaphore(CONCURRENCY)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 2)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        tasks = [fetch_one(sem, url, session) for url in pending]

        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            result = await coro

            if isinstance(result, Paper):
                ckpt.papers.append(result)
                ckpt.done_urls.add(result.url)
                log.info(
                    "[%d/%d] OK %s | %s",
                    done_count + i, total,
                    result.title[:80] if result.title else "(no title)",
                    result.publication_date or "(no date)",
                )
            else:
                # result is an error dict
                ckpt.errors.append(result)
                ckpt.done_urls.add(result["url"])
                batch_errors.append(result)
                log.warning(
                    "[%d/%d] ERR %s - %s",
                    done_count + i, total,
                    result["url"], result["error"],
                )

            # Periodic checkpoint
            if i % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(ckpt)
                save_csv(ckpt.papers)
                if batch_errors:
                    append_errors(batch_errors)
                    batch_errors = []

    # Step 4: final save
    save_checkpoint(ckpt)
    save_csv(ckpt.papers)
    if batch_errors:
        append_errors(batch_errors)

    _print_summary(ckpt.papers, total)


def _print_summary(papers: list[Paper], total: int) -> None:
    ok_count = sum(1 for p in papers if p.status == "ok")
    err_count = sum(1 for p in papers if p.status != "ok")
    log.info("=" * 55)
    log.info("DONE! %d / %d papers scraped successfully.", ok_count, total)
    log.info("Errors: %d", err_count)
    if ok_count > 0:
        log.info("Sample entries:")
        for p in papers[:3]:
            log.info("  - %s | %s | PDF: %s", p.title, p.publication_date, p.pdf_url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fresh = "--fresh" in sys.argv
    asyncio.run(scrape_all_async(fresh=fresh))


if __name__ == "__main__":
    main()
