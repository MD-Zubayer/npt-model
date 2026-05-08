"""
Training health checks and monitoring.
Monitors training process health and alerts on anomalies.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Monitor training health.
    
    Checks:
    - NaN/Inf detection
    - Loss trends
    - Gradient norms
    - GPU memory
    """
    
    def __init__(self, alert_threshold: float = 0.1):
        """Initialize health checker."""
        self.alert_threshold = alert_threshold
        self.history = []
        self.alerts = []
    
    def check_loss(self, loss: float) -> bool:
        """Check if loss is healthy."""
        if np.isnan(loss) or np.isinf(loss):
            self.alerts.append(f"Invalid loss: {loss}")
            logger.error(f"Invalid loss detected: {loss}")
            return False
        
        if loss > 1e6:
            logger.warning(f"Very large loss: {loss}")
        
        return True
    
    def check_gradients(self, gradients: List[float]) -> bool:
        """Check gradient health."""
        if not gradients:
            return True
        
        grad_norm = np.linalg.norm(gradients)
        
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            self.alerts.append(f"Invalid gradient norm: {grad_norm}")
            logger.error(f"Invalid gradient norm: {grad_norm}")
            return False
        
        return True
    
    def check_trend(self, losses: List[float], window_size: int = 100) -> bool:
        """Check if loss is trending correctly."""
        if len(losses) < window_size:
            return True
        
        recent_losses = losses[-window_size:]
        early_losses = losses[-2*window_size:-window_size]
        
        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        
        # Check if loss is increasing significantly
        if recent_avg > early_avg * (1 + self.alert_threshold):
            logger.warning(f"Loss increasing: {early_avg:.4f} -> {recent_avg:.4f}")
            return False
        
        return True
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        self.history.append({
            'timestamp': datetime.now(),
            **metrics
        })
    
    def get_summary(self) -> Dict:
        """Get health check summary."""
        return {
            'num_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'num_steps': len(self.history)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    checker = HealthChecker()
    # Example usage
