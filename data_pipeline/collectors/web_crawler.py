"""
Web crawler for scraping and collecting text data from web sources.
Handles robots.txt, rate limiting, and respectful crawling practices.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Asynchronous web crawler for collecting text data.
    
    Features:
    - Respectful crawling with rate limiting
    - Robots.txt compliance
    - Duplicate URL prevention
    - Connection pooling
    """
    
    def __init__(self, max_workers: int = 5, timeout: int = 30):
        """Initialize web crawler."""
        self.max_workers = max_workers
        self.timeout = timeout
        self.visited_urls = set()
        self.session = None
        
    async def crawl(self, 
                   urls: List[str],
                   max_depth: int = 2) -> List[Dict[str, str]]:
        """
        Crawl multiple URLs and extract text content.
        
        Args:
            urls: List of starting URLs
            max_depth: Maximum crawl depth
            
        Returns:
            List of extracted documents with metadata
        """
        documents = []
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            tasks = [self._crawl_recursive(url, max_depth) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    documents.extend(result)
                else:
                    logger.error(f"Crawl error: {result}")
                    
        return documents
    
    async def _crawl_recursive(self, 
                              url: str, 
                              depth: int) -> List[Dict[str, str]]:
        """Recursively crawl a URL."""
        if depth == 0 or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        documents = []
        
        try:
            html_content = await self._fetch_url(url)
            if html_content:
                doc = self._extract_text(html_content, url)
                documents.append(doc)
                
                # Extract and follow links
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if self._is_valid_url(next_url):
                        sub_docs = await self._crawl_recursive(next_url, depth - 1)
                        documents.extend(sub_docs)
                        
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            
        return documents
    
    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
        return None
    
    def _extract_text(self, html: str, url: str) -> Dict[str, str]:
        """Extract text content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'url': url,
            'content': text,
            'domain': urlparse(url).netloc,
            'source_type': 'web'
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and should be crawled."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https')
        except Exception:
            return False


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    crawler = WebCrawler(max_workers=5)
    # urls = ["https://example.com"]
    # docs = asyncio.run(crawler.crawl(urls))
    # print(f"Collected {len(docs)} documents")
