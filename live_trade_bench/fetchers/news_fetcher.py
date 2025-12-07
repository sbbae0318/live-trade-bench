from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

from bs4 import BeautifulSoup

from live_trade_bench.fetchers.base_fetcher import BaseFetcher
from live_trade_bench.utils.datetime_utils import parse_utc_datetime
# Simple in-memory cache: key = f"{query}|{start_date}|{end_date}" -> sorted news list
_NEWS_CACHE: Dict[str, List[Dict[str, Any]]] = {}


class NewsFetcher(BaseFetcher):
    def __init__(self, min_delay: float = 2.0, max_delay: float = 6.0):
        super().__init__(min_delay, max_delay)

    def _normalize_date(
        self, s: str, fallback_now: Optional[datetime] = None
    ) -> tuple[str, datetime]:
        """
            Normalize various date/time strings to ("MM/DD/YYYY", datetime) for Google News queries.
            Google News only supports date-level granularity, but we preserve the full datetime
            for post-filtering by time if needed.
        """
        if fallback_now is None:
            fallback_now = datetime.now()
        try:
            dt = parse_utc_datetime(s)
            # Google News URL parameter only accepts date format (MM/DD/YYYY)
            # Time component is preserved in dt for post-filtering
            return dt.strftime("%m/%d/%Y"), dt
        except Exception:
            return fallback_now.strftime("%m/%d/%Y"), fallback_now

    def _parse_relative_or_absolute(self, text: str, ref: datetime) -> float:
        t = text.strip().lower()

        # English patterns: "5 days ago", "1 hour ago"
        m = re.match(r"^\s*(\d+)\s+(second|minute|hour|day)s?\s+ago\s*$", t)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            delta = {
                "second": timedelta(seconds=num),
                "minute": timedelta(minutes=num),
                "hour": timedelta(hours=num),
                "day": timedelta(days=num),
            }[unit]
            return (ref - delta).timestamp()

        # Polish patterns: "5 dni temu", "16 godzin temu", "dzień temu"
        # Handle special case "dzień temu" (day ago) without number
        if t == "dzień temu":
            return (ref - timedelta(days=1)).timestamp()
        if t == "godzinę temu":
            return (ref - timedelta(hours=1)).timestamp()

        m = re.match(
            r"^\s*(\d+)\s+(sekund[ya]?|minut[ya]?|godzin[ya]?|dni)\s+temu\s*$", t
        )
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            # Map Polish units to timedelta
            if unit.startswith("sekund"):
                delta = timedelta(seconds=num)
            elif unit.startswith("minut"):
                delta = timedelta(minutes=num)
            elif unit.startswith("godzin"):
                delta = timedelta(hours=num)
            elif unit == "dni":
                delta = timedelta(days=num)
            else:
                delta = timedelta(0)
            return (ref - delta).timestamp()

        # Korean patterns: "9시간 전", "30분 전", "1일 전", "방금 전", "어제"
        # Normalize unicode spaces and remove surrounding spaces
        kt = re.sub(r"\s+", "", text)
        # Specific words
        if kt in ("방금전", "바로전"):
            return (ref - timedelta(seconds=30)).timestamp()
        if kt == "어제":
            return (ref - timedelta(days=1)).timestamp()
        m = re.match(r"^(\d+)(초|분|시간|일|주|개월?)전$", kt)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            if unit == "초":
                delta = timedelta(seconds=num)
            elif unit == "분":
                delta = timedelta(minutes=num)
            elif unit == "시간":
                delta = timedelta(hours=num)
            elif unit == "일":
                delta = timedelta(days=num)
            elif unit == "주":
                delta = timedelta(weeks=num)
            elif unit in ("개월", "달"):
                # Approximate month as 30 days
                delta = timedelta(days=30 * num)
            else:
                delta = timedelta(0)
            return (ref - delta).timestamp()

        # Absolute date formats
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text.strip(), fmt).timestamp()
            except ValueError:
                continue
        return ref.timestamp()

    def _clean_google_href(self, href: str) -> str:
        if href.startswith("/url?"):
            qs = parse_qs(urlparse(href).query)
            if "q" in qs and qs["q"]:
                return qs["q"][0]
        return href

    def _extract_snippet_from_url(self, url: str) -> str:
        try:
            # Add small delay to avoid rate limiting
            import time

            time.sleep(0.5)

            # Fetch the article page with timeout
            resp = self.make_request(url, timeout=10)
            if resp.status_code != 200:
                return ""

            soup = BeautifulSoup(resp.text, "html.parser")

            # Try to find article content in common containers
            content_selectors = [
                "article p",
                "[itemprop='articleBody'] p",
                ".article-content p",
                ".post-content p",
                "main p",
                "p",
            ]

            for selector in content_selectors:
                paragraphs = soup.select(selector)
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # Return first paragraph with substantial content (>50 chars)
                    if len(text) > 50:
                        # Limit snippet to 300 chars
                        return text[:300] if len(text) > 300 else text

            return ""

        except Exception:
            # Silently fail - snippet extraction is optional
            return ""

    def _find_snippet_dynamically(
        self, card, title_text, source_text, date_text
    ) -> str:
        candidates = []
        for div in card.find_all(["div", "span"]):
            text = div.get_text(strip=True)
            # Filter out empty or too short text
            if not text or len(text) < 20:
                continue

            # Filter out text that is exactly the title, source, or date
            if text in [title_text, source_text, date_text]:
                continue

            # Filter out text that contains the title (parent containers)
            if title_text in text and len(text) < len(title_text) + 50:
                continue

            candidates.append(text)

        # Return the longest candidate that remains
        if candidates:
            return max(candidates, key=len)

        return ""

    def fetch(
        self, query: str, start_date: str, end_date: str, max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        start_fmt, start_dt = self._normalize_date(start_date)
        end_fmt, end_dt = self._normalize_date(end_date)
        ref_date = end_dt  # Use end_date as reference for relative time parsing
        
        # Check if time component is specified (not 00:00:00)
        has_time_filter = (
            start_dt.hour != 0 or start_dt.minute != 0 or start_dt.second != 0 or
            end_dt.hour != 0 or end_dt.minute != 0 or end_dt.second != 0
        )

        # Use HTML-specific headers for Google News scraping
        html_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",  # Removed 'br' - brotli package not installed
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Maximum pages to fetch if filtering results in empty results
        max_total_pages = 20  # Limit to prevent infinite fetching
        
        results: List[Dict[str, Any]] = []
        current_max_pages = max_pages
        
        while len(results) == 0 and current_max_pages <= max_total_pages:
            # Fetch pages
            for page in range(current_max_pages):
                # URL-encode the query to handle spaces and special characters
                encoded_query = quote_plus(query)
                url = (
                    f"https://www.google.com/search?q={encoded_query}"
                    f"&tbs=cdr:1,cd_min:{start_fmt},cd_max:{end_fmt}"
                    f"&tbm=nws&start={page * 10}"
                )
                try:
                    resp = self.make_request(url, headers=html_headers, timeout=15)
                    # Use resp.text instead of resp.content to handle gzip encoding properly
                    soup = BeautifulSoup(resp.text, "html.parser")
                except Exception as e:
                    print(f"Request/parse failed: {e}")
                    break

                cards = soup.select("div.SoaBEf")
                if not cards:
                    break

                for el in cards:
                    try:
                        a = el.find("a")
                        if not a or "href" not in a.attrs:
                            continue
                        link = self._clean_google_href(a["href"])

                        title_el = el.select_one("div.MBeuO")
                        date_el = el.select_one(".LfVVr")
                        source_el = el.select_one(".NUnG9d span")

                        # Snippet is optional - title, date, and source are required
                        if not (title_el and date_el and source_el):
                            continue

                        # Get snippet with dynamic fallback strategy
                        snippet = self._find_snippet_dynamically(
                            el,
                            title_el.get_text(strip=True),
                            source_el.get_text(strip=True),
                            date_el.get_text(strip=True),
                        )

                        if not snippet and link and page == 0:
                            snippet = self._extract_snippet_from_url(link)

                        ts = self._parse_relative_or_absolute(
                            date_el.get_text(strip=True), ref_date
                        )

                        # Filter by time range if start_date or end_date have time components
                        # Google News only supports date-level filtering, so we filter client-side by timestamp
                        if has_time_filter:
                            article_ts = ts
                            start_ts = start_dt.timestamp()
                            end_ts = end_dt.timestamp()
                            
                            if article_ts < start_ts or article_ts > end_ts:
                                continue

                        results.append(
                            {
                                "link": link,
                                "title": title_el.get_text(strip=True),
                                "snippet": snippet,
                                "date": ts,
                                "source": source_el.get_text(strip=True),
                            }
                        )
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue

                if not soup.find("a", id="pnnext"):
                    break
            
            # If time filtering is active and no results after filtering, fetch more pages
            if has_time_filter and len(results) == 0 and current_max_pages < max_total_pages:
                # Increase pages for next iteration
                current_max_pages = min(current_max_pages + 5, max_total_pages)
                # Reset results to try again with more pages
                results = []
            else:
                # Either no time filter, or we have results, or reached max pages
                break

        return results


def fetch_news_data(
    query: str,
    start_date: str,
    end_date: str,
    max_pages: int = 1,
    ticker: Optional[str] = None,
    target_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fetcher = NewsFetcher()
    print(
        f"  - News fetcher with query '{query}' and start_date '{start_date}' and end_date '{end_date}'"
    )

    cache_key = f"{query}|{start_date}|{end_date}"

    if cache_key in _NEWS_CACHE:
        sorted_news = _NEWS_CACHE[cache_key]
    else:
        news_items = fetcher.fetch(query, start_date, end_date, max_pages)
        valid_news = [it for it in news_items if it.get("date") is not None]

        if target_date and valid_news:
            try:
                target_ts = parse_utc_datetime(target_date).timestamp()
                sorted_news = sorted(valid_news, key=lambda x: abs(x["date"] - target_ts))
            except Exception:
                sorted_news = sorted(valid_news, key=lambda x: x["date"], reverse=True)
        else:
            sorted_news = sorted(valid_news, key=lambda x: x["date"], reverse=True)

        _NEWS_CACHE[cache_key] = sorted_news

    if ticker:
        # Do not mutate cached entries; add tag on a shallow-copied list
        return [{**it, "tag": ticker} for it in sorted_news]
    return sorted_news
