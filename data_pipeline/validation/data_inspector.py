"""
Data validation and inspection module.
Checks data format, quality, and consistency.
"""

import logging
from typing import List, Dict, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataInspector:
    """
    Inspect and validate data quality and format.
    
    Features:
    - Format validation
    - Missing value checks
    - Schema validation
    - Statistical analysis
    - Consistency verification
    """
    
    def __init__(self, schema: Dict = None):
        """
        Initialize data inspector.
        
        Args:
            schema: Expected data schema
        """
        self.schema = schema or {
            'content': str,
            'source_type': str,
            'timestamp': str
        }
        
    def inspect_document(self, doc: Dict) -> Tuple[bool, List[str]]:
        """
        Inspect a single document.
        
        Args:
            doc: Document to inspect
            
        Returns:
            (is_valid, issues_list) tuple
        """
        issues = []
        
        # Check required fields
        for field in self.schema:
            if field not in doc:
                issues.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in self.schema.items():
            if field in doc and not isinstance(doc[field], expected_type):
                issues.append(f"Field '{field}' has wrong type: {type(doc[field]).__name__}")
        
        # Check content non-empty
        if 'content' in doc and not doc['content'].strip():
            issues.append("Content is empty")
        
        return len(issues) == 0, issues
    
    def inspect_batch(self, documents: List[Dict]) -> Dict:
        """
        Inspect a batch of documents.
        
        Args:
            documents: List of documents to inspect
            
        Returns:
            Statistics and issues
        """
        stats = {
            'total_documents': len(documents),
            'valid_documents': 0,
            'invalid_documents': 0,
            'common_issues': {},
            'avg_content_length': 0,
            'issues_by_document': []
        }
        
        total_length = 0
        
        for doc in documents:
            is_valid, issues = self.inspect_document(doc)
            
            if is_valid:
                stats['valid_documents'] += 1
            else:
                stats['invalid_documents'] += 1
            
            if issues:
                stats['issues_by_document'].append({
                    'document': doc.get('content', '')[:100],
                    'issues': issues
                })
                
                for issue in issues:
                    stats['common_issues'][issue] = stats['common_issues'].get(issue, 0) + 1
            
            if 'content' in doc:
                total_length += len(doc['content'])
        
        if documents:
            stats['avg_content_length'] = total_length / len(documents)
        
        return stats
    
    def validate_manifest(self, manifest_path: str) -> bool:
        """
        Validate dataset manifest file.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            True if valid
        """
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            required_fields = ['version', 'datasets', 'created_at']
            for field in required_fields:
                if field not in manifest:
                    logger.error(f"Missing manifest field: {field}")
                    return False
            
            logger.info("Manifest validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Manifest validation failed: {e}")
            return False
    
    def generate_report(self, documents: List[Dict]) -> str:
        """
        Generate validation report.
        
        Args:
            documents: List of documents
            
        Returns:
            Report string
        """
        stats = self.inspect_batch(documents)
        
        report = f"""
        === Data Validation Report ===
        Total Documents: {stats['total_documents']}
        Valid Documents: {stats['valid_documents']}
        Invalid Documents: {stats['invalid_documents']}
        Average Content Length: {stats['avg_content_length']:.0f} characters
        
        Common Issues:
        """
        
        for issue, count in stats['common_issues'].items():
            report += f"\n  - {issue}: {count} documents"
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    inspector = DataInspector()
    # Example usage
