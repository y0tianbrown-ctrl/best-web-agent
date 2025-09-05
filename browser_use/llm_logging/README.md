# LLM Logging Module

This module provides functionality to log LLM input messages and responses to files for debugging and analysis purposes.

## Features

- **Automatic Logging**: Captures all LLM interactions when enabled
- **Structured Output**: Logs are saved as JSON files with detailed information
- **Configurable**: Can be enabled/disabled via environment variables or code
- **Step Tracking**: Automatically tracks step numbers and timing
- **Token Usage**: Includes token usage information when available
- **Metadata Support**: Allows adding custom metadata to logs

## Configuration

### Environment Variables

Set these environment variables to configure LLM logging:

```bash
# Enable/disable LLM logging (default: false)
export BROWSER_USE_LLM_LOGGING_ENABLED=true

# Set the logging directory (default: llm_logs)
export BROWSER_USE_LLM_LOGGING_DIR=./my_llm_logs
```

### Programmatic Configuration

```python
from browser_use.llm_logging import set_llm_logging_enabled, set_llm_logging_directory

# Enable logging
set_llm_logging_enabled(True)

# Set custom directory
set_llm_logging_directory("./custom_logs")
```

## Usage

### Basic Usage

The logging is automatically enabled when you set the environment variable or call the configuration functions. No additional code is needed in your agent scripts.

```python
import os
from browser_use import Agent, ChatOpenAI

# Enable logging via environment variable
os.environ["BROWSER_USE_LLM_LOGGING_ENABLED"] = "true"

# Your normal agent code
llm = ChatOpenAI(model="gpt-4o")
agent = Agent(task="Your task here", llm=llm)
history = await agent.run()
```

### Advanced Usage

You can also use the logging service directly for more control:

```python
from browser_use.llm_logging import LLMLoggingService, get_llm_logging_service

# Get the global service
service = get_llm_logging_service()

# Check if logging is enabled
if service.enabled:
    print(f"Logs are saved to: {service.log_dir}")

# Get list of log files
log_files = service.get_log_files()
print(f"Found {len(log_files)} log files")

# Get the latest log
latest_log = service.get_latest_log()
if latest_log:
    print(f"Latest log: {latest_log['timestamp']}")

# Clear all logs
service.clear_logs()
```

## Log File Format

Each log file contains a JSON object with the following structure:

```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "model": "gpt-4o",
  "step": 1,
  "input": {
    "messages": [
      {
        "type": "SystemMessage",
        "content": "You are a helpful assistant.",
        "cache": false
      },
      {
        "type": "UserMessage", 
        "content": "Hello, how are you?",
        "cache": false
      }
    ],
    "message_count": 2
  },
  "response": {
    "type": "ChatInvokeCompletion",
    "completion": "I'm doing well, thank you for asking!",
    "usage": {
      "prompt_tokens": 15,
      "completion_tokens": 8,
      "total_tokens": 23
    },
    "raw_data": {
      "completion": "I'm doing well, thank you for asking!",
      "usage": "..."
    }
  },
  "metadata": {
    "task": "Search for Python programming information",
    "model": "gpt-4o",
    "step_number": 1,
    "timestamp": "2025-01-15T10:30:45.123456"
  }
}
```

### Field Descriptions

- **timestamp**: ISO format timestamp of the interaction
- **model**: Name of the LLM model used
- **step**: Step number from the agent execution
- **input**: 
  - **messages**: Array of input messages with type, content, and cache status
  - **message_count**: Total number of input messages
- **response**:
  - **type**: Type of the response object
  - **completion**: The actual completion text
  - **usage**: Token usage information (if available)
  - **raw_data**: Complete raw response data for debugging
- **metadata**: Additional information including task, model, step number, and timestamp

## File Naming

Log files are named using the pattern: `{model}_{timestamp}_step_{step}.json`

Examples:
- `gpt-4o_20250115_103045_123_step_1.json`
- `claude-3-sonnet_20250115_103045_456.json` (no step number)

## Examples

See `examples/llm_logging_example.py` and `examples/simple_llm_logging.py` for complete working examples.

## API Reference

### LLMLoggingService

Main service class for LLM logging.

#### Methods

- `log_llm_interaction(model, input_messages, response, step=None, metadata=None)`: Log an LLM interaction
- `set_enabled(enabled)`: Enable or disable logging
- `get_log_files()`: Get list of all log files
- `get_latest_log()`: Get the most recent log file content
- `clear_logs()`: Clear all log files

### Global Functions

- `get_llm_logging_service()`: Get the global logging service instance
- `set_llm_logging_enabled(enabled)`: Enable or disable logging globally
- `set_llm_logging_directory(log_dir)`: Set the logging directory globally

## Troubleshooting

### Logging Not Working

1. Check that `BROWSER_USE_LLM_LOGGING_ENABLED` is set to `true`
2. Verify the logging directory is writable
3. Check that the agent is properly initialized

### Large Log Files

- Log files can become large with long conversations
- Consider periodically clearing old logs using `service.clear_logs()`
- Monitor disk space usage in the logging directory

### Performance Impact

- Logging adds minimal overhead to LLM calls
- File I/O is synchronous but won't block the main execution
- Disable logging in production if not needed

## Integration Details

The logging is automatically integrated into the `Agent.get_model_output()` method, which is called every time the agent makes an LLM request. This ensures that all LLM interactions are captured without requiring any changes to your existing code.

The logging happens after the LLM response is received but before it's processed, ensuring that both successful and failed interactions are captured.
