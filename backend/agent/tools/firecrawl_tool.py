import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "firecrawl.log")

os.makedirs(LOG_DIR, exist_ok=True)


logger = logging.getLogger("firecrawl_tool")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BASE_URL = "https://api.firecrawl.dev"

DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_CRAWL_TIMEOUT = 60

MAX_DEPTH = 3
MAX_LIMIT = 50

MAX_POLL_ATTEMPTS = 15
POLL_INTERVAL_SECONDS = 8

MAX_ERROR_BODY_LENGTH = 500


# ============================================================
# ERROR CODES
# ============================================================

class ErrorCode:
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INVALID_URL = "INVALID_URL"

    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"

    HTTP_ERROR = "HTTP_ERROR"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"

    CRAWL_START_FAILED = "CRAWL_START_FAILED"
    CRAWL_FAILED = "CRAWL_FAILED"
    CRAWL_TIMEOUT = "CRAWL_TIMEOUT"
    CRAWL_NO_DATA = "CRAWL_NO_DATA"

    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class FirecrawlError(Exception):
    """
    Custom exception containing structured error information.
    """

    def __init__(
        self,
        message: str,
        code: str = ErrorCode.UNKNOWN_ERROR,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)

        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


# ============================================================
# FIRECRAWL TOOL
# ============================================================

class FirecrawlTool:
    """
    Production-oriented Firecrawl API wrapper.

    Features:
    - URL validation
    - API key validation
    - Retry mechanism
    - Timeout handling
    - Structured errors
    - Detailed logging
    - Crawl polling
    - Single-page fallback
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.base_url = base_url.rstrip("/")

        self.session = requests.Session()

        self._configure_session()

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if not self.api_key:
            logger.error(
                "Firecrawl API key is missing. "
                "Set FIRECRAWL_API_KEY in environment variables."
            )

    # ========================================================
    # SESSION / RETRY CONFIGURATION
    # ========================================================

    def _configure_session(self) -> None:
        """
        Configure HTTP connection pooling and retries.
        """

        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ========================================================
    # PUBLIC METHOD
    # ========================================================

    def scrape_website(
        self,
        url: str,
        include_links: bool = True,
        max_depth: int = 3,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Scrape a website.

        Attempts:
        1. Validate configuration
        2. Validate URL
        3. Crawl multiple pages if requested
        4. Fall back to single page scraping if crawl fails
        """

        request_id = str(uuid.uuid4())

        logger.info(
            "[%s] Starting website scraping | url=%s | "
            "include_links=%s | max_depth=%s | limit=%s",
            request_id,
            url,
            include_links,
            max_depth,
            limit,
        )

        try:

            # ------------------------------------------------
            # Configuration validation
            # ------------------------------------------------

            self._validate_configuration()

            # ------------------------------------------------
            # URL validation
            # ------------------------------------------------

            normalized_url = self._normalize_url(url)

            # ------------------------------------------------
            # Parameter validation
            # ------------------------------------------------

            max_depth = self._validate_depth(max_depth)
            limit = self._validate_limit(limit)

            # ------------------------------------------------
            # Crawl
            # ------------------------------------------------

            if include_links and max_depth > 0:

                logger.info(
                    "[%s] Starting multi-page crawl",
                    request_id,
                )

                result = self._crawl_multiple_pages(
                    normalized_url,
                    max_depth,
                    limit,
                    request_id,
                )

                if result.get("success"):
                    logger.info(
                        "[%s] Multi-page crawl completed successfully",
                        request_id,
                    )

                    return result

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                logger.warning(
                    "[%s] Multi-page crawl failed. "
                    "Falling back to single-page scrape.",
                    request_id,
                )

            # ------------------------------------------------
            # Single-page scraping
            # ------------------------------------------------

            logger.info(
                "[%s] Performing single-page scrape",
                request_id,
            )

            return self._scrape_single_page_simple(
                normalized_url,
                request_id,
            )

        except FirecrawlError as exc:

            logger.error(
                "[%s] Firecrawl error | code=%s | message=%s",
                request_id,
                exc.code,
                exc.message,
            )

            return self._error_response(
                error=exc,
                request_id=request_id,
                url=url,
            )

        except Exception as exc:

            logger.exception(
                "[%s] Unexpected error while scraping website",
                request_id,
            )

            error = FirecrawlError(
                message=str(exc),
                code=ErrorCode.UNKNOWN_ERROR,
                details={
                    "exception_type": type(exc).__name__,
                },
            )

            return self._error_response(
                error=error,
                request_id=request_id,
                url=url,
            )

    # ========================================================
    # CONFIGURATION VALIDATION
    # ========================================================

    def _validate_configuration(self) -> None:

        if not self.api_key:

            raise FirecrawlError(
                message="Firecrawl API key is not configured.",
                code=ErrorCode.CONFIGURATION_ERROR,
            )

        if not self.base_url:

            raise FirecrawlError(
                message="Firecrawl base URL is empty.",
                code=ErrorCode.CONFIGURATION_ERROR,
            )

    # ========================================================
    # URL VALIDATION
    # ========================================================

    def _normalize_url(self, url: str) -> str:

        if not url or not isinstance(url, str):

            raise FirecrawlError(
                message="URL must be a non-empty string.",
                code=ErrorCode.INVALID_URL,
            )

        url = url.strip()

        if not url:
            raise FirecrawlError(
                message="URL cannot be empty.",
                code=ErrorCode.INVALID_URL,
            )

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise FirecrawlError(
                message=f"Unsupported URL scheme: {parsed.scheme}",
                code=ErrorCode.INVALID_URL,
            )

        if not parsed.netloc:
            raise FirecrawlError(
                message="URL does not contain a valid hostname.",
                code=ErrorCode.INVALID_URL,
            )

        return url

    # ========================================================
    # PARAMETER VALIDATION
    # ========================================================

    def _validate_depth(self, max_depth: int) -> int:

        if not isinstance(max_depth, int):
            raise FirecrawlError(
                message="max_depth must be an integer.",
                code=ErrorCode.INVALID_URL,
            )

        if max_depth < 0:
            raise FirecrawlError(
                message="max_depth cannot be negative.",
                code=ErrorCode.INVALID_URL,
            )

        return min(max_depth, MAX_DEPTH)

    def _validate_limit(self, limit: int) -> int:

        if not isinstance(limit, int):
            raise FirecrawlError(
                message="limit must be an integer.",
                code=ErrorCode.INVALID_URL,
            )

        if limit <= 0:
            raise FirecrawlError(
                message="limit must be greater than zero.",
                code=ErrorCode.INVALID_URL,
            )

        return min(limit, MAX_LIMIT)

    # ========================================================
    # SINGLE PAGE SCRAPING
    # ========================================================

    def _scrape_single_page_simple(
        self,
        url: str,
        request_id: str,
    ) -> Dict[str, Any]:

        endpoint = f"{self.base_url}/v0/scrape"

        scrape_config = {
            "url": url,
            "formats": ["markdown"],
            "waitFor": 2000,
            "timeout": 15000,
        }

        logger.info(
            "[%s] Sending single-page scrape request | endpoint=%s",
            request_id,
            endpoint,
        )

        try:

            response = self.session.post(
                endpoint,
                headers=self.headers,
                json=scrape_config,
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )

        except requests.exceptions.Timeout as exc:

            logger.exception(
                "[%s] Firecrawl scrape request timed out",
                request_id,
            )

            return self._failure(
                code=ErrorCode.TIMEOUT_ERROR,
                message="Firecrawl API request timed out.",
                request_id=request_id,
                url=url,
            )

        except requests.exceptions.ConnectionError as exc:

            logger.exception(
                "[%s] Firecrawl connection failed",
                request_id,
            )

            return self._failure(
                code=ErrorCode.CONNECTION_ERROR,
                message="Unable to connect to Firecrawl API.",
                request_id=request_id,
                url=url,
            )

        except requests.exceptions.RequestException as exc:

            logger.exception(
                "[%s] Firecrawl request failed",
                request_id,
            )

            return self._failure(
                code=ErrorCode.NETWORK_ERROR,
                message="Firecrawl network request failed.",
                request_id=request_id,
                url=url,
            )

        # ----------------------------------------------------
        # HTTP STATUS
        # ----------------------------------------------------

        logger.info(
            "[%s] Firecrawl response | status=%s",
            request_id,
            response.status_code,
        )

        if not response.ok:

            error_body = self._safe_response_text(response)

            logger.error(
                "[%s] Firecrawl HTTP error | status=%s | body=%s",
                request_id,
                response.status_code,
                error_body,
            )

            return self._failure(
                code=ErrorCode.HTTP_ERROR,
                message=(
                    f"Firecrawl returned HTTP "
                    f"{response.status_code}."
                ),
                request_id=request_id,
                url=url,
                status_code=response.status_code,
            )

        # ----------------------------------------------------
        # JSON PARSING
        # ----------------------------------------------------

        result = self._parse_json_response(
            response,
            request_id,
        )

        if result is None:

            return self._failure(
                code=ErrorCode.INVALID_RESPONSE,
                message="Firecrawl returned invalid JSON.",
                request_id=request_id,
                url=url,
                status_code=response.status_code,
            )

        # ----------------------------------------------------
        # FIRECRAWL SUCCESS
        # ----------------------------------------------------

        if not result.get("success", False):

            error_message = result.get(
                "error",
                "Unknown Firecrawl API error.",
            )

            logger.error(
                "[%s] Firecrawl API returned failure | error=%s",
                request_id,
                error_message,
            )

            return self._failure(
                code=ErrorCode.API_ERROR,
                message=str(error_message),
                request_id=request_id,
                url=url,
                status_code=response.status_code,
            )

        # ----------------------------------------------------
        # CONTENT EXTRACTION
        # ----------------------------------------------------

        try:

            content = self._extract_content(result)

        except Exception as exc:

            logger.exception(
                "[%s] Failed to extract Firecrawl content",
                request_id,
            )

            return self._failure(
                code=ErrorCode.EXTRACTION_ERROR,
                message="Failed to extract scraped content.",
                request_id=request_id,
                url=url,
            )

        if not content.strip():

            logger.warning(
                "[%s] Firecrawl succeeded but returned empty content",
                request_id,
            )

            return self._failure(
                code=ErrorCode.INVALID_RESPONSE,
                message="Firecrawl returned empty content.",
                request_id=request_id,
                url=url,
            )

        logger.info(
            "[%s] Single-page scraping successful | content_length=%s",
            request_id,
            len(content),
        )

        return {
            "success": True,
            "content": content,
            "metadata": result.get("metadata", {}),
            "url": url,
            "request_id": request_id,
        }

    # ========================================================
    # MULTI-PAGE CRAWLING
    # ========================================================

    def _crawl_multiple_pages(
        self,
        url: str,
        max_depth: int,
        limit: int,
        request_id: str,
    ) -> Dict[str, Any]:

        endpoint = f"{self.base_url}/v0/crawl"

        crawl_config = {
            "url": url,
            "crawlerOptions": {
                "maxDepth": max_depth,
                "limit": limit,
                "allowBackwardCrawling": False,
                "allowExternalContent": False,
            },
            "pageOptions": {
                "formats": ["markdown"],
                "waitFor": 2000,
                "timeout": 10000,
            },
        }

        logger.info(
            "[%s] Starting crawl | depth=%s | limit=%s",
            request_id,
            max_depth,
            limit,
        )

        try:

            response = self.session.post(
                endpoint,
                headers=self.headers,
                json=crawl_config,
                timeout=DEFAULT_CRAWL_TIMEOUT,
            )

        except requests.exceptions.Timeout:

            logger.exception(
                "[%s] Crawl start request timed out",
                request_id,
            )

            return self._failure(
                code=ErrorCode.TIMEOUT_ERROR,
                message="Crawl request timed out.",
                request_id=request_id,
                url=url,
            )

        except requests.exceptions.ConnectionError:

            logger.exception(
                "[%s] Crawl connection failed",
                request_id,
            )

            return self._failure(
                code=ErrorCode.CONNECTION_ERROR,
                message="Unable to connect to Firecrawl.",
                request_id=request_id,
                url=url,
            )

        except requests.exceptions.RequestException:

            logger.exception(
                "[%s] Crawl request failed",
                request_id,
            )

            return self._failure(
                code=ErrorCode.NETWORK_ERROR,
                message="Crawl network request failed.",
                request_id=request_id,
                url=url,
            )

        logger.info(
            "[%s] Crawl start response | status=%s",
            request_id,
            response.status_code,
        )

        if not response.ok:

            logger.error(
                "[%s] Crawl start failed | status=%s | body=%s",
                request_id,
                response.status_code,
                self._safe_response_text(response),
            )

            return self._failure(
                code=ErrorCode.CRAWL_START_FAILED,
                message=(
                    f"Crawl start failed with HTTP "
                    f"{response.status_code}."
                ),
                request_id=request_id,
                url=url,
                status_code=response.status_code,
            )

        crawl_result = self._parse_json_response(
            response,
            request_id,
        )

        if not crawl_result:

            return self._failure(
                code=ErrorCode.INVALID_RESPONSE,
                message="Invalid JSON from crawl start request.",
                request_id=request_id,
                url=url,
            )

        job_id = crawl_result.get("jobId")

        if not job_id:

            logger.error(
                "[%s] Crawl response did not contain jobId | response_keys=%s",
                request_id,
                list(crawl_result.keys()),
            )

            return self._failure(
                code=ErrorCode.CRAWL_START_FAILED,
                message="Firecrawl did not return a crawl job ID.",
                request_id=request_id,
                url=url,
            )

        logger.info(
            "[%s] Crawl job started | job_id=%s",
            request_id,
            job_id,
        )

        return self._poll_crawl_results(
            job_id=job_id,
            original_url=url,
            request_id=request_id,
        )

    # ========================================================
    # POLL CRAWL
    # ========================================================

    def _poll_crawl_results(
        self,
        job_id: str,
        original_url: str,
        request_id: str,
    ) -> Dict[str, Any]:

        endpoint = f"{self.base_url}/v0/crawl/status/{job_id}"

        logger.info(
            "[%s] Polling crawl job | job_id=%s",
            request_id,
            job_id,
        )

        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):

            try:

                response = self.session.get(
                    endpoint,
                    headers=self.headers,
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                )

            except requests.exceptions.Timeout:

                logger.exception(
                    "[%s] Crawl status request timed out | "
                    "attempt=%s/%s",
                    request_id,
                    attempt,
                    MAX_POLL_ATTEMPTS,
                )

                # Don't immediately fail the entire crawl.
                # Try again.
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            except requests.exceptions.RequestException:

                logger.exception(
                    "[%s] Crawl status request failed | "
                    "attempt=%s/%s",
                    request_id,
                    attempt,
                    MAX_POLL_ATTEMPTS,
                )

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if not response.ok:

                logger.error(
                    "[%s] Crawl status HTTP error | status=%s | "
                    "attempt=%s/%s",
                    request_id,
                    response.status_code,
                    attempt,
                    MAX_POLL_ATTEMPTS,
                )

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            result = self._parse_json_response(
                response,
                request_id,
            )

            if result is None:

                logger.error(
                    "[%s] Invalid JSON in crawl status response",
                    request_id,
                )

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            status = str(
                result.get("status", "")
            ).lower()

            logger.info(
                "[%s] Crawl status=%s | attempt=%s/%s",
                request_id,
                status,
                attempt,
                MAX_POLL_ATTEMPTS,
            )

            # ------------------------------------------------
            # COMPLETED
            # ------------------------------------------------

            if status == "completed":

                pages = result.get("data", [])

                if not isinstance(pages, list):

                    logger.error(
                        "[%s] Crawl completed but data is not a list",
                        request_id,
                    )

                    return self._failure(
                        code=ErrorCode.INVALID_RESPONSE,
                        message="Invalid crawl data format.",
                        request_id=request_id,
                        url=original_url,
                    )

                if not pages:

                    logger.warning(
                        "[%s] Crawl completed but returned no pages",
                        request_id,
                    )

                    return self._failure(
                        code=ErrorCode.CRAWL_NO_DATA,
                        message="Crawl completed but no pages were returned.",
                        request_id=request_id,
                        url=original_url,
                    )

                try:

                    combined_content = (
                        self._combine_crawled_content(pages)
                    )

                except Exception:

                    logger.exception(
                        "[%s] Failed to combine crawled pages",
                        request_id,
                    )

                    return self._failure(
                        code=ErrorCode.EXTRACTION_ERROR,
                        message="Failed to combine crawled content.",
                        request_id=request_id,
                        url=original_url,
                    )

                if not combined_content.strip():

                    logger.warning(
                        "[%s] Crawl returned pages but no usable content",
                        request_id,
                    )

                    return self._failure(
                        code=ErrorCode.CRAWL_NO_DATA,
                        message="Crawled pages contained no usable content.",
                        request_id=request_id,
                        url=original_url,
                    )

                logger.info(
                    "[%s] Crawl completed successfully | pages=%s | "
                    "content_length=%s",
                    request_id,
                    len(pages),
                    len(combined_content),
                )

                return {
                    "success": True,
                    "content": combined_content,
                    "metadata": {
                        "pages_crawled": len(pages),
                        "crawl_depth": result.get(
                            "crawlDepth",
                            1,
                        ),
                    },
                    "url": original_url,
                    "request_id": request_id,
                }

            # ------------------------------------------------
            # FAILED
            # ------------------------------------------------

            if status in ("failed", "error"):

                error_message = result.get(
                    "error",
                    "Firecrawl crawl job failed.",
                )

                logger.error(
                    "[%s] Crawl job failed | job_id=%s | error=%s",
                    request_id,
                    job_id,
                    error_message,
                )

                return self._failure(
                    code=ErrorCode.CRAWL_FAILED,
                    message=str(error_message),
                    request_id=request_id,
                    url=original_url,
                )

            # ------------------------------------------------
            # UNKNOWN STATUS
            # ------------------------------------------------

            if status not in (
                "queued",
                "scraping",
                "crawling",
                "processing",
                "pending",
                "completed",
                "failed",
                "error",
            ):

                logger.warning(
                    "[%s] Unknown crawl status=%s",
                    request_id,
                    status,
                )

            # ------------------------------------------------
            # WAIT
            # ------------------------------------------------

            if attempt < MAX_POLL_ATTEMPTS:

                time.sleep(POLL_INTERVAL_SECONDS)

        # ----------------------------------------------------
        # POLLING TIMEOUT
        # ----------------------------------------------------

        logger.error(
            "[%s] Crawl polling exceeded maximum attempts | "
            "job_id=%s | attempts=%s",
            request_id,
            job_id,
            MAX_POLL_ATTEMPTS,
        )

        return self._failure(
            code=ErrorCode.CRAWL_TIMEOUT,
            message=(
                "Crawl did not complete within the maximum "
                "polling period."
            ),
            request_id=request_id,
            url=original_url,
        )

    # ========================================================
    # CONTENT EXTRACTION
    # ========================================================

    def _extract_content(
        self,
        result: Dict[str, Any],
    ) -> str:

        content_parts = []

        markdown_content = result.get("markdown")

        if isinstance(markdown_content, str):
            markdown_content = markdown_content.strip()

            if markdown_content:
                content_parts.append(markdown_content)

        html_content = result.get("html")

        if (
            isinstance(html_content, str)
            and html_content.strip()
            and not content_parts
        ):

            cleaned_html = self._clean_html_content(
                html_content
            )

            if cleaned_html:
                content_parts.append(cleaned_html)

        metadata = result.get("metadata")

        if isinstance(metadata, dict):

            title = metadata.get("title")
            description = metadata.get("description")

            if isinstance(title, str) and title.strip():

                content_parts.insert(
                    0,
                    f"Title: {title.strip()}",
                )

            if (
                isinstance(description, str)
                and description.strip()
            ):

                insert_position = (
                    1
                    if title
                    else 0
                )

                content_parts.insert(
                    insert_position,
                    f"Description: {description.strip()}",
                )

        final_content = "\n\n".join(content_parts).strip()

        logger.debug(
            "Content extraction completed | characters=%s",
            len(final_content),
        )

        return final_content

    # ========================================================
    # COMBINE CRAWLED PAGES
    # ========================================================

    def _combine_crawled_content(
        self,
        pages: list,
    ) -> str:

        combined_parts = []

        for index, page in enumerate(pages):

            if not isinstance(page, dict):

                logger.warning(
                    "Skipping invalid page at index=%s",
                    index,
                )

                continue

            markdown = page.get("markdown")
            html = page.get("html")

            if not markdown and not html:
                continue

            page_url = page.get(
                "url",
                f"Page {index + 1}",
            )

            page_content = []

            page_content.append(
                f"=== {page_url} ==="
            )

            if isinstance(markdown, str) and markdown.strip():

                content = markdown.strip()

            elif isinstance(html, str) and html.strip():

                content = self._clean_html_content(html)

            else:

                content = ""

            if content:
                page_content.append(content)

            if len(page_content) > 1:

                combined_parts.append(
                    "\n".join(page_content)
                )

        if not combined_parts:
            return ""

        return (
            "\n\n"
            + "=" * 50
            + "\n\n"
            .join(combined_parts)
        )

    # ========================================================
    # HTML CLEANING
    # ========================================================

    def _clean_html_content(
        self,
        html_content: str,
    ) -> str:

        if not isinstance(html_content, str):
            return ""

        clean_text = re.sub(
            r"<[^>]+>",
            " ",
            html_content,
        )

        clean_text = re.sub(
            r"\s+",
            " ",
            clean_text,
        )

        return clean_text.strip()

    # ========================================================
    # JSON RESPONSE
    # ========================================================

    def _parse_json_response(
        self,
        response: requests.Response,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:

        try:

            data = response.json()

        except ValueError:

            logger.error(
                "[%s] Response was not valid JSON | "
                "status=%s | body=%s",
                request_id,
                response.status_code,
                self._safe_response_text(response),
            )

            return None

        if not isinstance(data, dict):

            logger.error(
                "[%s] Expected JSON object but received %s",
                request_id,
                type(data).__name__,
            )

            return None

        return data

    # ========================================================
    # SAFE RESPONSE BODY
    # ========================================================

    def _safe_response_text(
        self,
        response: requests.Response,
    ) -> str:

        try:

            text = response.text or ""

            return text[
                :MAX_ERROR_BODY_LENGTH
            ].replace("\n", " ")

        except Exception:

            return "<unable to read response body>"

    # ========================================================
    # FAILURE RESPONSE
    # ========================================================

    def _failure(
        self,
        code: str,
        message: str,
        request_id: str,
        url: str,
        status_code: Optional[int] = None,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "error": message,
            "error_code": code,
            "content": "",
            "metadata": {},
            "url": url,
            "request_id": request_id,
            "status_code": status_code,
        }

    # ========================================================
    # EXCEPTION RESPONSE
    # ========================================================

    def _error_response(
        self,
        error: FirecrawlError,
        request_id: str,
        url: str,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "error": error.message,
            "error_code": error.code,
            "content": "",
            "metadata": {},
            "url": url,
            "request_id": request_id,
            "status_code": error.status_code,
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self) -> None:

        try:

            self.session.close()

            logger.info(
                "Firecrawl HTTP session closed"
            )

        except Exception:

            logger.exception(
                "Failed to close Firecrawl HTTP session"
            )

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()
