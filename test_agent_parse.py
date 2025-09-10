#!/usr/bin/env python3
"""
Test AgentOutput.model_validate_json() with your JSON string
"""

import json
from pydantic import ValidationError

# Your JSON string - Fixed to include required 'down' field
json_string = '''{
  "memory": "Scrolled down to load search results for 'Interestellar'.",
  "action": [
    {
      "scroll": {
        "num_pages": 1.2
      }
    }
  ]
}'''

def test_agent_output_validation():
    """Test AgentOutput.model_validate_json with your JSON string"""
    try:
        # Import required modules
        from browser_use.agent.views import AgentOutput
        from browser_use.tools.service import Tools
        
        print("✅ Successfully imported modules")
        
        # Create tools instance to get the proper ActionModel
        tools = Tools()
        
        # Create the proper ActionModel with scroll action
        action_model = tools.registry.create_action_model()
        
        # Create AgentOutput with the proper ActionModel
        agent_output_class = AgentOutput.type_with_custom_actions(action_model)
        
        print("✅ Successfully created AgentOutput class with scroll action")
        
        # Validate your JSON string
        agent_output = agent_output_class.model_validate_json(json_string)
        
        print("✅ SUCCESS! Validation passed")
        print(f"Memory: {agent_output.memory}")
        print(f"Number of actions: {len(agent_output.action)}")
        
        # Process the actions
        for i, action in enumerate(agent_output.action):
            print(f"\nAction {i+1}:")
            print(f"  Type: {type(action).__name__}")
            print(f"  Data: {action.model_dump(exclude_unset=True)}")
            
            # Check for scroll action
            if hasattr(action, 'scroll') and action.scroll:
                scroll = action.scroll
                print(f"  Scroll details:")
                print(f"    - Pages: {scroll.num_pages}")
                print(f"    - Direction: {'down' if scroll.down else 'up'}")
                print(f"    - Frame index: {scroll.frame_element_index}")
        
        return agent_output
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    print("Testing AgentOutput.model_validate_json()")
    print("=" * 50)
    
    result = test_agent_output_validation()
    
    if result:
        print("\n✅ AgentOutput.model_validate_json() works perfectly!")
    else:
        print("\n❌ There was an issue. Check the error messages above.")