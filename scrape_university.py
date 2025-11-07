import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

HEADERS = {"User-Agent": "StudentAssistant/0.1 (+mailto:******@gmail.com)"}

def scrapify(url):
    """Fetch HTML content from a URL."""
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text

def extract_text(html):
    """Extract text from HTML, removing scripts/styles/nav/footer/header."""
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "nav", "footer", "header"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def crawl(start_url, allowed_domain=None, max_pages=50, delay=1.0, max_depth=2):
    """
    Crawl the start_url, following links within the same domain up to max_depth levels.
    """
    if allowed_domain is None:
        allowed_domain = urlparse(start_url).netloc

    seen = set()
    queue = [(start_url, 0)]  
    out_dir = "scraped_pages"
    os.makedirs(out_dir, exist_ok=True)

    while queue and len(seen) < max_pages:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue

        print(f"Fetching (depth {depth}): {url}")
        try:
            html = scrapify(url)
        except Exception as e:
            print("Error:", e)
            continue

        text = extract_text(html)
        fname = os.path.join(out_dir, f"page_{len(seen):04d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(url + "\n\n" + text)
        seen.add(url)

        # find links on same domain
        if depth < max_depth:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                p = urlparse(href)
                if p.netloc.endswith(allowed_domain):
                    if href not in seen and all(href != q[0] for q in queue):
                        queue.append((href, depth + 1))

        time.sleep(delay)

    print("Done. Saved", len(seen), "pages in", out_dir)
    return out_dir

if __name__ == "__main__":
    start_url = "https://www.inf.ovgu.de/inf/en/Study/Before+you+start+studies/Study+courses/Master+courses/Data+and+Knowledge+Engineering.html"
    crawl(start_url, max_pages=40, delay=1.0, max_depth=2)
