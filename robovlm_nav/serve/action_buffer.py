"""
Action Buffer for Chunk Reuse Strategy

Manages storage and retrieval of action chunks for efficient inference.
"""

import numpy as np
from typing import Optional
import threading


class ActionBuffer:
    """
    Thread-safe action buffer for chunk reuse strategy
    
    Stores action chunks (typically 10 actions) and provides 
    sequential access for fast inference.
    """
    
    def __init__(self, chunk_size: int = 10):
        """
        Initialize action buffer
        
        Args:
            chunk_size: Number of actions in each chunk (default: 10)
        """
        self.chunk_size = chunk_size
        self.buffer = []
        self.lock = threading.Lock()
    
    def push_chunk(self, actions: np.ndarray) -> None:
        """
        Store a new action chunk in the buffer
        
        Args:
            actions: Action chunk of shape (N, action_dim) where N <= chunk_size
        """
        with self.lock:
            if len(actions.shape) == 1:
                actions = actions.reshape(1, -1)
            
            # Convert to list of individual actions
            for action in actions:
                self.buffer.append(action.copy())
    
    def pop_action(self) -> Optional[np.ndarray]:
        """
        Get the next action from the buffer
        
        Returns:
            Next action as numpy array, or None if buffer is empty
        """
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer.pop(0)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty and needs refill"""
        with self.lock:
            return len(self.buffer) == 0
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all buffered actions"""
        with self.lock:
            self.buffer.clear()
    
    def get_status(self) -> dict:
        """Get buffer status for debugging"""
        with self.lock:
            return {
                "size": len(self.buffer),
                "capacity": self.chunk_size,
                "is_empty": len(self.buffer) == 0,
                "needs_refill": len(self.buffer) == 0
            }
