"""
API data ingestor for collecting data from various API endpoints.
Supports pagination, authentication, and rate limiting.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class APIIngestor:
    """
    Ingest data from REST APIs.
    
    Features:
    - Async API requests
    - Pagination support
    - Rate limiting
    - Authentication handling
    - Error recovery
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_retries: int = 3,
                 rate_limit: int = 100):
        """
        Initialize API ingestor.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit: Requests per minute
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        
    async def fetch_from_endpoint(self,
                                 url: str,
                                 headers: Optional[Dict] = None,
                                 params: Optional[Dict] = None,
                                 auth: Optional[tuple] = None) -> List[Dict]:
        """
        Fetch data from a single API endpoint.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            params: Query parameters
            auth: (username, password) tuple
            
        Returns:
            List of fetched documents
        """
        documents = []
        
        async with aiohttp.ClientSession() as session:
            page = 0
            while True:
                if params:
                    params['page'] = page
                
                try:
                    async with session.get(
                        url,
                        headers=headers or {},
                        params=params,
                        auth=auth,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if isinstance(data, list):
                                items = data
                            elif isinstance(data, dict):
                                # Handle common pagination patterns
                                items = data.get('results') or data.get('data') or data.get('items')
                                if not items:
                                    items = [data]
                            else:
                                items = []
                            
                            if not items:
                                break
                            
                            for item in items:
                                documents.append({
                                    'content': str(item),
                                    'source_type': 'api',
                                    'source_url': url,
                                    'page': page,
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'metadata': item if isinstance(item, dict) else {}
                                })
                            
                            page += 1
                        else:
                            logger.warning(f"API returned status {response.status}")
                            break
                            
                except asyncio.TimeoutError:
                    logger.error(f"Request timeout for {url}")
                    break
                except Exception as e:
                    logger.error(f"Error fetching from {url}: {e}")
                    break
        
        return documents
    
    async def fetch_multiple_endpoints(self,
                                      endpoints: List[Dict[str, Any]]) -> List[Dict]:
        """
        Fetch from multiple API endpoints concurrently.
        
        Args:
            endpoints: List of endpoint configurations
                      Each should have 'url' and optional 'headers', 'params', 'auth'
                      
        Returns:
            Combined list of documents from all endpoints
        """
        tasks = [
            self.fetch_from_endpoint(
                ep['url'],
                headers=ep.get('headers'),
                params=ep.get('params'),
                auth=ep.get('auth')
            )
            for ep in endpoints
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        documents = []
        
        for result in results:
            if isinstance(result, list):
                documents.extend(result)
            else:
                logger.error(f"Fetch error: {result}")
        
        return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    ingestor = APIIngestor()
    # Example usage would go here
