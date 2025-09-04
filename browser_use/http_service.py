from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from browser_use import Agent, ChatOpenAI
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.browser.profile import ViewportSize

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Browser-Use HTTP Service",
    description="Simplified HTTP service for browser automation with real XPath selectors",
    version="2.0.0"
)

SELECTOR_PROMPT = """
IMPORTANT! For each action, specify the selector type and value:
- "xpathSelector" for XPath expressions
- Return actions as JSON with selector type "xpathSelector" and value element.xpath
"""

class ActionInfo(BaseModel):
    selector: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    text: Optional[str] = None
    value: Optional[str] = None
    up: Optional[bool] = None
    down: Optional[bool] = None
    
    def dict(self, **kwargs):
        """Return dictionary representation excluding None values"""
        data = super().dict(**kwargs)
        return {k: v for k, v in data.items() if v is not None}

class TaskRequest(BaseModel):
    instructions: str = Field(..., description="Task instructions for the browser agent")
    model: str = Field("gpt-4o", description="LLM model to use")
    timeout: Optional[int] = Field(300, description="Task timeout in seconds")
    headless: Optional[bool] = Field(False, description="Run browser in headless mode")

class TaskResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    actions: Optional[List[Dict[str, Any]]] = None
    total_actions: Optional[int] = None
    timing: Optional[Dict[str, float]] = None

@app.get("/")
async def root():
    return {"message": "Browser-Use HTTP Service with Real XPath Selectors", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "browser-use"}

async def extract_real_xpath_actions(history, browser_session) -> List[ActionInfo]:
    """Extract actions with real XPath selectors from DOM elements"""
    actions = []
    
    try:
        if hasattr(history, 'action_history'):
            action_history = history.action_history()
            for action in (action_history or []):
                action_info = await _parse_action_with_real_xpath(action, browser_session)
                if action_info:
                    actions.append(action_info)
        else:
            history_list = list(history) if hasattr(history, '__iter__') else []
            for item in history_list:
                action_info = await _parse_action_with_real_xpath(item, browser_session)
                if action_info:
                    actions.append(action_info)
    except Exception as e:
        logger.warning(f"Could not extract actions: {e}")
    
    return actions

async def _parse_action_with_real_xpath(action_item, browser_session) -> Optional[ActionInfo]:
    """Parse action and get real XPath from DOM element"""
    try:
        # Handle ActionModel objects (from runtime capture)
        if hasattr(action_item, 'model_dump'):
            action_data = action_item.model_dump(exclude_unset=True)
            action_name = next(iter(action_data.keys())) if action_data else None
            
            if action_name == 'click_element_by_index':
                element_index = action_data['click_element_by_index'].get('index')
                real_xpath = await _get_real_xpath_from_index(element_index, browser_session)
                return ActionInfo(
                    type="ClickAction",
                    selector={"type": "xpathSelector", "value": real_xpath}
                )
                
            elif action_name == 'input_text':
                element_index = action_data['input_text'].get('index')
                text = action_data['input_text'].get('text')
                real_xpath = await _get_real_xpath_from_index(element_index, browser_session)
                return ActionInfo(
                    type="TypeAction",
                    selector={"type": "xpathSelector", "value": real_xpath},
                    text=text
                )
                
            elif action_name == 'go_to_url':
                url = action_data['go_to_url']['url']
                return ActionInfo(
                    type="NavigateAction",
                    selector={"type": "xpathSelector", "value": f"//html[@data-url='{url}']"}
                )
                
            elif action_name == 'scroll':
                scroll_data = action_data['scroll']
                down = scroll_data.get('down', True)
                num_pages = scroll_data.get('num_pages', 1.0)
                up = not down
                return ActionInfo(
                    type="ScrollAction",
                    selector={},
                    value=str(num_pages),
                    up=up,
                    down=down
                )
        
        # Handle list format (from history extraction)
        elif isinstance(action_item, list) and len(action_item) > 0:
            action_data = action_item[0]
            if isinstance(action_data, dict):
                
                if 'click_element_by_index' in action_data:
                    element_index = action_data['click_element_by_index'].get('index')
                    real_xpath = await _get_real_xpath_from_index(element_index, browser_session)
                    return ActionInfo(
                        type="ClickAction",
                        selector={"type": "xpathSelector", "value": real_xpath}
                    )
                    
                elif 'input_text' in action_data:
                    element_index = action_data['input_text'].get('index')
                    text = action_data['input_text'].get('text')
                    real_xpath = await _get_real_xpath_from_index(element_index, browser_session)
                    return ActionInfo(
                        type="TypeAction",
                        selector={"type": "xpathSelector", "value": real_xpath},
                        text=text
                    )
                    
                elif 'go_to_url' in action_data:
                    url = action_data['go_to_url']['url']
                    return ActionInfo(
                        type="NavigateAction",
                        selector={"type": "xpathSelector", "value": f"//html[@data-url='{url}']"}
                    )
                    
                elif 'scroll' in action_data:
                    scroll_data = action_data['scroll']
                    down = scroll_data.get('down', True)
                    num_pages = scroll_data.get('num_pages', 1.0)
                    up = not down
                    return ActionInfo(
                        type="ScrollAction",
                        selector={},
                        value=str(num_pages),
                        up=up,
                        down=down
                    )
    except Exception as e:
        logger.debug(f"Could not parse action: {e}")
    
    return None

async def _get_real_xpath_from_index(index: int, browser_session) -> str:
    """Get real XPath from browser-use element index"""
    try:
        if browser_session:
            # Method 1: Try get_dom_element_by_index (cached)
            element = await browser_session.get_dom_element_by_index(index)
            if element and hasattr(element, 'xpath') and element.xpath:
                logger.debug(f"Got real XPath for index {index}: {element.xpath}")
                return element.xpath
            
            # Method 2: Try get_selector_map (multiple sources)
            selector_map = await browser_session.get_selector_map()
            if selector_map and index in selector_map:
                element = selector_map[index]
                if hasattr(element, 'xpath') and element.xpath:
                    logger.debug(f"Got XPath from selector map for index {index}: {element.xpath}")
                    return element.xpath
                
                # Generate XPath from element attributes
                if hasattr(element, 'attributes') and element.attributes:
                    xpath = _generate_xpath_from_attributes(element)
                    logger.debug(f"Generated XPath from selector map for index {index}: {xpath}")
                    return xpath
            
            # Method 3: Try browser state summary (most comprehensive)
            try:
                # Check if browser session has proper event handlers
                if hasattr(browser_session, 'event_bus') and browser_session.event_bus:
                    browser_state = await browser_session.get_browser_state_summary(
                        cache_clickable_elements_hashes=True,
                        include_screenshot=False,
                        cached=False
                    )
                    if browser_state and browser_state.dom_state and browser_state.dom_state.selector_map:
                        selector_map = browser_state.dom_state.selector_map
                        if index in selector_map:
                            element = selector_map[index]
                            if hasattr(element, 'xpath') and element.xpath:
                                logger.debug(f"Got XPath from browser state for index {index}: {element.xpath}")
                                return element.xpath
                            
                            # Generate XPath from element attributes
                            if hasattr(element, 'attributes') and element.attributes:
                                xpath = _generate_xpath_from_attributes(element)
                                logger.debug(f"Generated XPath from browser state for index {index}: {xpath}")
                                return xpath
                else:
                    logger.debug(f"Browser session has no event bus, skipping browser state method for index {index}")
            except Exception as e:
                logger.debug(f"Could not get browser state for index {index}: {e}")
                # Don't let this error propagate, continue to fallback
                
    except Exception as e:
        logger.debug(f"Could not get real XPath for index {index}: {e}")
    
    # Final fallback - position-based selector
    logger.warning(f"Could not find element for index {index}, using fallback selector")
    return f"//*[position()={index}]"

def _generate_xpath_from_attributes(element) -> str:
    """Generate real XPath from actual DOM element attributes"""
    try:
        tag = element.node_name.lower() if hasattr(element, 'node_name') else 'div'
        attrs = element.attributes or {}
        
        # Priority 1: ID (most reliable)
        if attrs.get('id'):
            return f"//{tag}[@id='{attrs['id']}']"
        
        # Priority 2: Name (for forms)
        if attrs.get('name'):
            return f"//{tag}[@name='{attrs['name']}']"
            
        # Priority 3: Class (use first class)
        if attrs.get('class'):
            class_value = attrs['class'].strip().split()[0]
            return f"//{tag}[contains(@class, '{class_value}')]"
            
        # Priority 4: Text content
        if hasattr(element, 'get_all_children_text'):
            text = element.get_all_children_text(max_depth=1).strip()
            if text and len(text) < 50:
                return f"//{tag}[contains(text(), '{text}')]"
        
        # Priority 5: Type attribute (for inputs)
        if attrs.get('type'):
            return f"//{tag}[@type='{attrs['type']}']"
        
        # Fallback: just tag name
        return f"//{tag}"
        
    except Exception:
        return "//div"

@app.post("/run-task", response_model=TaskResponse)
async def run_task(req: TaskRequest):
    """Execute a browser automation task with real XPath selectors"""
    # Initialize timing measurements
    timing = {}
    start_time = time.time()
    
    try:
        logger.info(f"Running task: {req.instructions[:50]}...")
        
        # Time LLM creation
        llm_start = time.time()
        llm = ChatOpenAI(model=req.model)
        timing['llm_creation'] = time.time() - llm_start
        
        # Time browser session creation
        browser_start = time.time()
        browser_profile = BrowserProfile(
            headless=req.headless,
            viewport=ViewportSize(width=1280, height=720),
            stealth=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"] if req.headless else []
        )
        
        browser_session = BrowserSession(browser_profile=browser_profile)
        timing['browser_session_creation'] = time.time() - browser_start
        
        # Store captured actions during execution
        captured_actions = []
        
        # Define callback to capture actions during execution
        async def capture_actions_callback(browser_state, agent_output, step_number):
            """Capture actions as they happen during execution"""
            try:
                if hasattr(agent_output, 'action') and agent_output.action:
                    for action in agent_output.action:
                        action_info = await _parse_action_with_real_xpath(action, browser_session)
                        if action_info:
                            captured_actions.append(action_info)
                            logger.debug(f"Captured action in step {step_number}: {action_info.type}")
            except Exception as e:
                logger.debug(f"Could not capture action in step {step_number}: {e}")
        
        # Time agent creation
        agent_start = time.time()
        agent = Agent(
            task=req.instructions,
            llm=llm,
            browser_session=browser_session,
            register_new_step_callback=capture_actions_callback,
            flash_mode=True  # Enable flash mode for faster execution
        )
        timing['agent_creation'] = time.time() - agent_start
        
        # Time agent execution (main task)
        execution_start = time.time()
        history = await asyncio.wait_for(agent.run(), timeout=req.timeout)
        timing['agent_execution'] = time.time() - execution_start
        
        # Time action extraction
        extraction_start = time.time()
        if captured_actions:
            actions = captured_actions
            logger.info(f"Used {len(actions)} actions captured during execution")
        else:
            # Fallback: Extract actions from history (less reliable)
            actions = await extract_real_xpath_actions(history, agent.browser_session)
            logger.info(f"Fallback: Extracted {len(actions)} actions from history")
        timing['action_extraction'] = time.time() - extraction_start
        
        # Time result processing
        result_start = time.time()
        result = "Task completed successfully"
        if history:
            history_list = list(history) if hasattr(history, '__iter__') else []
            if history_list:
                last_item = history_list[-1]
                if hasattr(last_item, 'content') and not isinstance(last_item, tuple):
                    result = str(last_item.content)
                else:
                    result = str(last_item)
        
        # Clean actions by removing None values
        clean_actions = []
        for action in actions:
            if action:
                clean_actions.append(action.dict())
        
        timing['result_processing'] = time.time() - result_start
        
        # Calculate total time and add timing analysis
        total_time = time.time() - start_time
        timing['total_time'] = total_time
        
        # Add percentage breakdown (create new dict to avoid modification during iteration)
        percentage_timing = {}
        for key, value in timing.items():
            if key != 'total_time' and total_time > 0:
                percentage_timing[f'{key}_percentage'] = round((value / total_time) * 100, 2)
        
        # Merge percentage timing into main timing dict
        timing.update(percentage_timing)
        
        logger.info(f"Task completed with {len(actions)} actions in {total_time:.2f}s")
        logger.info(f"Timing breakdown: {timing}")
        
        return TaskResponse(
            success=True,
            result=result,
            actions=clean_actions,
            total_actions=len(actions),
            timing=timing
        )
        
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        timing['total_time'] = total_time
        timing['error'] = 'timeout'
        logger.error(f"Task timed out after {total_time:.2f}s")
        raise HTTPException(status_code=408, detail=f"Task timed out after {total_time:.2f}s")
    except Exception as e:
        total_time = time.time() - start_time
        timing['total_time'] = total_time
        timing['error'] = str(e)
        logger.error(f"Task failed after {total_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Browser-Use HTTP Service on 127.0.0.1:9000")
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")