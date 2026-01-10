"""
Memory Block Manager - Utilities for managing Letta agent memory blocks.
"""

from typing import Dict, List, Any, Optional
import json


class MemoryBlockManager:
    """
    Utility class for managing Letta agent memory blocks.
    
    Memory blocks are persistent context sections that the agent
    can read and update. This class provides higher-level operations
    for common memory patterns.
    """
    
    def __init__(self, client, agent_id: str):
        """
        Initialize memory manager.
        
        Args:
            client: Letta client instance
            agent_id: Agent ID to manage
        """
        self.client = client
        self.agent_id = agent_id
    
    def get_block(self, label: str) -> Optional[Dict]:
        """
        Get memory block by label.
        
        Args:
            label: Block label (e.g., "experiment_log")
        
        Returns:
            Block dict or None if not found
        """
        blocks = list(self.client.agents.blocks.list(self.agent_id))
        return next((b for b in blocks if b.label == label), None)
    
    def get_block_value(self, label: str, default: Any = None) -> Any:
        """
        Get parsed value from memory block.
        
        Args:
            label: Block label
            default: Default value if block doesn't exist
        
        Returns:
            Parsed JSON value or default
        """
        block = self.get_block(label)
        if not block:
            return default
        
        try:
            return json.loads(block.value)
        except json.JSONDecodeError:
            return block.value  # Return as string if not JSON
    
    def set_block_value(self, label: str, value: Any):
        """
        Set memory block value (creates if doesn't exist).
        
        Args:
            label: Block label
            value: Value to set (will be JSON serialized if dict/list)
        """
        # Serialize if needed
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, indent=2)
        else:
            value_str = str(value)
        
        block = self.get_block(label)
        
        if block:
            # Update existing
            self.client.agents.blocks.update(
                self.agent_id,
                block.id,
                value=value_str
            )
        else:
            # Create new
            self.client.agents.blocks.create(
                self.agent_id,
                label=label,
                value=value_str
            )
    
    def append_to_list_block(
        self,
        label: str,
        item: Any,
        max_items: Optional[int] = None
    ):
        """
        Append item to a list-based memory block.
        
        Args:
            label: Block label
            item: Item to append
            max_items: Maximum items to keep (FIFO if exceeded)
        """
        current_list = self.get_block_value(label, default=[])
        
        if not isinstance(current_list, list):
            current_list = []
        
        current_list.append(item)
        
        # Trim if needed
        if max_items and len(current_list) > max_items:
            current_list = current_list[-max_items:]
        
        self.set_block_value(label, current_list)
    
    def list_all_blocks(self) -> List[Dict[str, Any]]:
        """
        List all memory blocks for the agent.
        
        Returns:
            List of block info dicts
        """
        blocks = list(self.client.agents.blocks.list(self.agent_id))
        return [
            {
                "label": block.label,
                "id": block.id,
                "size": len(block.value),
                "preview": block.value[:100] + "..." if len(block.value) > 100 else block.value
            }
            for block in blocks
        ]
