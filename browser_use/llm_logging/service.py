"""
LLM Logging Service

This service captures LLM input messages and responses to files for debugging and analysis purposes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from browser_use.config import CONFIG
from browser_use.llm.messages import BaseMessage


class LLMLoggingService:
    """Service for logging LLM input messages and responses to files"""
    
    def __init__(self, log_dir: str | None = None, enabled: bool | None = None):
        """
        Initialize the LLM logging service
        
        Args:
            log_dir: Directory to store log files (defaults to config)
            enabled: Whether logging is enabled (defaults to config)
        """
        # Use config values as defaults
        self.log_dir = Path(log_dir or CONFIG.BROWSER_USE_LLM_LOGGING_DIR)
        self.enabled = enabled if enabled is not None else CONFIG.BROWSER_USE_LLM_LOGGING_ENABLED
        
        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
    
    def _serialize_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Serialize messages to a JSON-serializable format"""
        serialized = []
        for msg in messages:
            # Handle content serialization properly
            content = self._serialize_content(msg.content)
            
            msg_dict = {
                "type": type(msg).__name__,
                "content": content,
            }
            
            # Add additional fields if they exist
            if hasattr(msg, 'name') and msg.name:
                msg_dict["name"] = msg.name
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                msg_dict["additional_kwargs"] = msg.additional_kwargs
            if hasattr(msg, 'cache') and msg.cache is not None:
                msg_dict["cache"] = msg.cache
                
            serialized.append(msg_dict)
        
        return serialized
    
    def _serialize_content(self, content: Any) -> Any:
        """Serialize message content to JSON-serializable format"""
        if content is None:
            return None
        elif isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle list of content parts (e.g., multimodal messages)
            serialized_parts = []
            for part in content:
                if hasattr(part, 'model_dump'):
                    # Pydantic model
                    serialized_parts.append(part.model_dump())
                elif isinstance(part, dict):
                    # Dictionary - keep as is
                    serialized_parts.append(part)
                elif hasattr(part, '__dict__'):
                    # Object with attributes - serialize recursively
                    serialized_parts.append(self._serialize_content(part))
                else:
                    # Simple value
                    serialized_parts.append(str(part))
            return serialized_parts
        elif isinstance(content, dict):
            # Dictionary - keep as is
            return content
        elif hasattr(content, 'model_dump'):
            # Pydantic model
            return content.model_dump()
        elif hasattr(content, '__dict__'):
            # Object with attributes - convert to JSON-serializable dict
            try:
                # Try to serialize the object's attributes recursively
                obj_dict = {}
                for k, v in content.__dict__.items():
                    if v is None:
                        obj_dict[k] = None
                    elif isinstance(v, (str, int, float, bool)):
                        obj_dict[k] = v
                    elif isinstance(v, list):
                        # Recursively serialize list items
                        obj_dict[k] = [self._serialize_content(item) for item in v]
                    elif isinstance(v, dict):
                        # Recursively serialize dict values
                        obj_dict[k] = {key: self._serialize_content(val) for key, val in v.items()}
                    elif hasattr(v, '__dict__'):
                        obj_dict[k] = self._serialize_content(v)
                    else:
                        obj_dict[k] = str(v)
                return obj_dict
            except Exception:
                # Fallback to string representation if serialization fails
                return {k: str(v) for k, v in content.__dict__.items()}
        else:
            # Fallback to string representation
            return str(content)
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """Serialize LLM response to JSON-serializable format"""
        response_data = {
            "type": type(response).__name__,
        }
        
        # Handle completion
        if hasattr(response, 'completion'):
            response_data["completion"] = self._serialize_content(response.completion)
        else:
            response_data["completion"] = str(response)
        
        # Handle usage information
        if hasattr(response, 'usage') and response.usage:
            usage_data = {}
            if hasattr(response.usage, 'prompt_tokens'):
                usage_data["prompt_tokens"] = response.usage.prompt_tokens
            if hasattr(response.usage, 'completion_tokens'):
                usage_data["completion_tokens"] = response.usage.completion_tokens
            if hasattr(response.usage, 'total_tokens'):
                usage_data["total_tokens"] = response.usage.total_tokens
            response_data["usage"] = usage_data
        else:
            response_data["usage"] = None
        
        # Add raw response data for debugging
        if hasattr(response, 'model_dump'):
            response_data["raw_data"] = response.model_dump()
        elif hasattr(response, '__dict__'):
            response_data["raw_data"] = {k: str(v) for k, v in response.__dict__.items()}
        
        return response_data
    
    def _get_log_filename(self, model: str, step: Optional[int] = None) -> str:
        """Generate log filename based on model and step"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        step_suffix = f"_step_{step}" if step is not None else ""
        return f"{model}_{timestamp}{step_suffix}.json"
    
    def log_llm_interaction(
        self,
        model: str,
        input_messages: List[BaseMessage],
        response: Any,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log an LLM interaction to a file
        
        Args:
            model: Name of the LLM model
            input_messages: Input messages sent to the LLM
            response: Response from the LLM
            step: Step number (optional)
            metadata: Additional metadata to log
            
        Returns:
            Path to the log file if logging was successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            # Prepare the log data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "step": step,
                "input": {
                    "messages": self._serialize_messages(input_messages),
                    "message_count": len(input_messages)
                },
                "response": self._serialize_response(response)
            }
            
            # Add metadata if provided
            if metadata:
                log_data["metadata"] = metadata
            
            # Write to file
            filename = self._get_log_filename(model, step)
            filepath = self.log_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            return str(filepath)
            
        except Exception as e:
            # Don't raise exceptions in logging to avoid breaking the main flow
            print(f"Warning: Failed to log LLM interaction: {e}")
            return None
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable logging"""
        self.enabled = enabled
        if enabled:
            self.log_dir.mkdir(exist_ok=True)
    
    def get_log_files(self) -> List[Path]:
        """Get list of all log files"""
        if not self.log_dir.exists():
            return []
        return list(self.log_dir.glob("*.json"))
    
    def clear_logs(self) -> None:
        """Clear all log files"""
        if self.log_dir.exists():
            for file in self.log_dir.glob("*.json"):
                file.unlink()
    
    def get_latest_log(self) -> Optional[Dict[str, Any]]:
        """Get the most recent log file content"""
        log_files = self.get_log_files()
        if not log_files:
            return None
        
        # Sort by modification time, get the latest
        latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None


# Global instance
_llm_logging_service = LLMLoggingService()


def get_llm_logging_service() -> LLMLoggingService:
    """Get the global LLM logging service instance"""
    return _llm_logging_service


def set_llm_logging_enabled(enabled: bool) -> None:
    """Enable or disable LLM logging globally"""
    _llm_logging_service.set_enabled(enabled)


def set_llm_logging_directory(log_dir: str) -> None:
    """Set the LLM logging directory"""
    _llm_logging_service.log_dir = Path(log_dir)
    if _llm_logging_service.enabled:
        _llm_logging_service.log_dir.mkdir(exist_ok=True)
