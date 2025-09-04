"""Default browser action handlers using CDP."""

import asyncio
import json
import platform

from browser_use.browser.events import (
	ClickElementEvent,
	DoubleClickEvent,
	GetDropdownOptionsEvent,
	GoBackEvent,
	GoForwardEvent,
	HoverEvent,
	RefreshEvent,
	ScrollEvent,
	ScrollToTextEvent,
	SelectDropdownOptionEvent,
	SelectTextEvent,
	SendKeysEvent,
	SubmitFormEvent,
	TypeTextEvent,
	UploadFileEvent,
	WaitEvent,
)
from browser_use.browser.views import BrowserError, URLNotAllowedError
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.dom.service import EnhancedDOMTreeNode

# Import EnhancedDOMTreeNode and rebuild event models that have forward references to it
# This must be done after all imports are complete
ClickElementEvent.model_rebuild()
GetDropdownOptionsEvent.model_rebuild()
SelectDropdownOptionEvent.model_rebuild()
TypeTextEvent.model_rebuild()
ScrollEvent.model_rebuild()
UploadFileEvent.model_rebuild()
DoubleClickEvent.model_rebuild()
HoverEvent.model_rebuild()
SelectTextEvent.model_rebuild()
SubmitFormEvent.model_rebuild()


class DefaultActionWatchdog(BaseWatchdog):
	"""Handles default browser actions like click, type, and scroll using CDP."""

	async def on_ClickElementEvent(self, event: ClickElementEvent) -> dict | None:
		"""Handle click request with CDP."""
		try:
			# Check if session is alive before attempting any operations
			if not self.browser_session.agent_focus or not self.browser_session.agent_focus.target_id:
				error_msg = 'Cannot execute click: browser session is corrupted (target_id=None). Session may have crashed.'
				self.logger.error(f'⚠️ {error_msg}')
				raise BrowserError(error_msg)

			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'
			starting_target_id = self.browser_session.agent_focus.target_id

			# Track initial number of tabs to detect new tab opening
			initial_target_ids = await self.browser_session._cdp_get_all_pages()

			# Check if element is a file input (should not be clicked)
			if self.browser_session.is_file_input(element_node):
				msg = f'Index {index_for_logging} - has an element which opens file upload dialog. To upload files please use a specific function to upload files'
				self.logger.info(msg)
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			# Perform the actual click using internal implementation
			click_metadata = None
			click_metadata = await self._click_element_node_impl(element_node, while_holding_ctrl=event.while_holding_ctrl)
			download_path = None  # moved to downloads_watchdog.py

			# Build success message
			if download_path:
				msg = f'Downloaded file to {download_path}'
				self.logger.info(f'💾 {msg}')
			else:
				msg = f'Clicked button with index {index_for_logging}: {element_node.get_all_children_text(max_depth=2)}'
				self.logger.debug(f'🖱️ {msg}')
			self.logger.debug(f'Element xpath: {element_node.xpath}')

			# Wait a bit for potential new tab to be created
			# This is necessary because tab creation is async and might not be immediate
			await asyncio.sleep(0.1)

			# Note: We don't clear cached state here - let multi_act handle DOM change detection
			# by explicitly rebuilding and comparing when needed
			# Successfully clicked, always reset session back to parent page session context
			self.browser_session.agent_focus = await self.browser_session.get_or_create_cdp_session(
				target_id=starting_target_id, focus=True
			)

			# Check if a new tab was opened
			after_target_ids = await self.browser_session._cdp_get_all_pages()
			new_target_ids = {t['targetId'] for t in after_target_ids} - {t['targetId'] for t in initial_target_ids}
			new_tab_opened = len(new_target_ids) > 0

			if new_target_ids:
				new_tab_msg = 'New tab opened - switching to it'
				msg += f' - {new_tab_msg}'
				self.logger.info(f'🔗 {new_tab_msg}')

				if not event.while_holding_ctrl:
					# if while_holding_ctrl=False it means agent was not expecting a new tab to be opened
					# so we need to switch to the new tab to make the agent aware of the surprise new tab that was opened.
					# when while_holding_ctrl=True we dont actually want to switch to it,
					# we should match human expectations of ctrl+click which opens in the background,
					# so in multi_act it usually already sends [click_element_by_index(123, while_holding_ctrl=True), switch_tab(tab_id=None)] anyway
					from browser_use.browser.events import SwitchTabEvent

					new_target_id = new_target_ids.pop()
					switch_event = await self.event_bus.dispatch(SwitchTabEvent(target_id=new_target_id))
					await switch_event

			# Return click metadata including new tab information
			result_metadata = click_metadata if isinstance(click_metadata, dict) else {}
			result_metadata['new_tab_opened'] = new_tab_opened

			return result_metadata
		except Exception as e:
			raise

	async def on_TypeTextEvent(self, event: TypeTextEvent) -> dict | None:
		"""Handle text input request with CDP."""
		try:
			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Check if this is index 0 or a falsy index - type to the page (whatever has focus)
			if not element_node.element_index or element_node.element_index == 0:
				# Type to the page without focusing any specific element
				await self._type_to_page(event.text)
				self.logger.info(f'⌨️ Typed "{event.text}" to the page (current focus)')
				return None  # No coordinates available for page typing
			else:
				try:
					# Try to type to the specific element
					input_metadata = await self._input_text_element_node_impl(
						element_node, event.text, clear_existing=event.clear_existing or (not event.text)
					)
					self.logger.info(f'⌨️ Typed "{event.text}" into element with index {index_for_logging}')
					self.logger.debug(f'Element xpath: {element_node.xpath}')
					return input_metadata  # Return coordinates if available
				except Exception as e:
					# Element not found or error - fall back to typing to the page
					self.logger.warning(f'Failed to type to element {index_for_logging}: {e}. Falling back to page typing.')
					try:
						await asyncio.wait_for(self._click_element_node_impl(element_node, while_holding_ctrl=False), timeout=3.0)
					except Exception as e:
						pass
					await self._type_to_page(event.text)
					self.logger.info(f'⌨️ Typed "{event.text}" to the page as fallback')
					return None  # No coordinates available for fallback typing

			# Note: We don't clear cached state here - let multi_act handle DOM change detection
			# by explicitly rebuilding and comparing when needed
		except Exception as e:
			raise

	async def on_ScrollEvent(self, event: ScrollEvent) -> None:
		"""Handle scroll request with CDP."""
		# Check if we have a current target for scrolling
		if not self.browser_session.agent_focus:
			error_msg = 'No active target for scrolling'
			raise BrowserError(error_msg)

		try:
			# Convert direction and amount to pixels
			# Positive pixels = scroll down, negative = scroll up
			pixels = event.amount if event.direction == 'down' else -event.amount

			# CRITICAL: CDP calls time out without this, even if the target is already active
			await self.browser_session.agent_focus.cdp_client.send.Target.activateTarget(
				params={'targetId': self.browser_session.agent_focus.target_id}
			)

			# Element-specific scrolling if node is provided
			if event.node is not None:
				element_node = event.node
				index_for_logging = element_node.backend_node_id or 'unknown'

				# Check if the element is an iframe
				is_iframe = element_node.tag_name and element_node.tag_name.upper() == 'IFRAME'

				# Try to scroll the element's container
				success = await self._scroll_element_container(element_node, pixels)
				if success:
					self.logger.debug(
						f'📜 Scrolled element {index_for_logging} container {event.direction} by {event.amount} pixels'
					)

					# CRITICAL: For iframe scrolling, we need to force a full DOM refresh
					# because the iframe's content has changed position
					if is_iframe:
						self.logger.debug('🔄 Forcing DOM refresh after iframe scroll')
						# Note: We don't clear cached state here - let multi_act handle DOM change detection
						# by explicitly rebuilding and comparing when needed

						# Wait a bit for the scroll to settle and DOM to update
						await asyncio.sleep(0.2)

					return None

			# Perform target-level scroll
			await self._scroll_with_cdp_gesture(pixels)

			# CRITICAL: CDP calls time out without this, even if the target is already active
			await self.browser_session.agent_focus.cdp_client.send.Target.activateTarget(
				params={'targetId': self.browser_session.agent_focus.target_id}
			)

			# Note: We don't clear cached state here - let multi_act handle DOM change detection
			# by explicitly rebuilding and comparing when needed

			# Log success
			self.logger.debug(f'📜 Scrolled {event.direction} by {event.amount} pixels')
			return None
		except Exception as e:
			raise

	# ========== Implementation Methods ==========

	async def _click_element_node_impl(self, element_node, while_holding_ctrl: bool = False) -> dict | None:
		"""
		Click an element using pure CDP with multiple fallback methods for getting element geometry.

		Args:
			element_node: The DOM element to click
			new_tab: If True, open any resulting navigation in a new tab
		"""

		try:
			# Check if element is a file input or select dropdown - these should not be clicked
			tag_name = element_node.tag_name.lower() if element_node.tag_name else ''
			element_type = element_node.attributes.get('type', '').lower() if element_node.attributes else ''

			if tag_name == 'select':
				msg = f'Cannot click on <select> elements. Use get_dropdown_options(index={element_node.element_index}) action instead.'
				self.logger.warning(msg)
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			if tag_name == 'input' and element_type == 'file':
				msg = f'Cannot click on file input element (index={element_node.element_index}). File uploads must be handled using upload_file_to_element action.'
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			# Get CDP client
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get the correct session ID for the element's frame
			session_id = cdp_session.session_id

			# Get element bounds
			backend_node_id = element_node.backend_node_id

			# Get viewport dimensions for visibility checks
			layout_metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(session_id=session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Try multiple methods to get element geometry
			quads = []

			# Method 1: Try DOM.getContentQuads first (best for inline elements and complex layouts)
			try:
				content_quads_result = await cdp_session.cdp_client.send.DOM.getContentQuads(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				if 'quads' in content_quads_result and content_quads_result['quads']:
					quads = content_quads_result['quads']
					self.logger.debug(f'Got {len(quads)} quads from DOM.getContentQuads')
			except Exception as e:
				self.logger.debug(f'DOM.getContentQuads failed: {e}')

			# Method 2: Fall back to DOM.getBoxModel
			if not quads:
				try:
					box_model = await cdp_session.cdp_client.send.DOM.getBoxModel(
						params={'backendNodeId': backend_node_id}, session_id=session_id
					)
					if 'model' in box_model and 'content' in box_model['model']:
						content_quad = box_model['model']['content']
						if len(content_quad) >= 8:
							# Convert box model format to quad format
							quads = [
								[
									content_quad[0],
									content_quad[1],  # x1, y1
									content_quad[2],
									content_quad[3],  # x2, y2
									content_quad[4],
									content_quad[5],  # x3, y3
									content_quad[6],
									content_quad[7],  # x4, y4
								]
							]
							self.logger.debug('Got quad from DOM.getBoxModel')
				except Exception as e:
					self.logger.debug(f'DOM.getBoxModel failed: {e}')

			# Method 3: Fall back to JavaScript getBoundingClientRect
			if not quads:
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					if 'object' in result and 'objectId' in result['object']:
						object_id = result['object']['objectId']

						# Get bounding rect via JavaScript
						bounds_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
							params={
								'functionDeclaration': """
									function() {
										const rect = this.getBoundingClientRect();
										return {
											x: rect.left,
											y: rect.top,
											width: rect.width,
											height: rect.height
										};
									}
								""",
								'objectId': object_id,
								'returnByValue': True,
							},
							session_id=session_id,
						)

						if 'result' in bounds_result and 'value' in bounds_result['result']:
							rect = bounds_result['result']['value']
							# Convert rect to quad format
							x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
							quads = [
								[
									x,
									y,  # top-left
									x + w,
									y,  # top-right
									x + w,
									y + h,  # bottom-right
									x,
									y + h,  # bottom-left
								]
							]
							self.logger.debug('Got quad from getBoundingClientRect')
				except Exception as e:
					self.logger.debug(f'JavaScript getBoundingClientRect failed: {e}')

			# If we still don't have quads, fall back to JS click
			if not quads:
				self.logger.warning('⚠️ Could not get element geometry from any method, falling back to JavaScript click')
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.click(); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.05)
					# Navigation is handled by BrowserSession via events
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript click also failed: {js_e}')
					raise Exception(f'Failed to click element: {js_e}')

			# Find the largest visible quad within the viewport
			best_quad = None
			best_area = 0

			for quad in quads:
				if len(quad) < 8:
					continue

				# Calculate quad bounds
				xs = [quad[i] for i in range(0, 8, 2)]
				ys = [quad[i] for i in range(1, 8, 2)]
				min_x, max_x = min(xs), max(xs)
				min_y, max_y = min(ys), max(ys)

				# Check if quad intersects with viewport
				if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
					continue  # Quad is completely outside viewport

				# Calculate visible area (intersection with viewport)
				visible_min_x = max(0, min_x)
				visible_max_x = min(viewport_width, max_x)
				visible_min_y = max(0, min_y)
				visible_max_y = min(viewport_height, max_y)

				visible_width = visible_max_x - visible_min_x
				visible_height = visible_max_y - visible_min_y
				visible_area = visible_width * visible_height

				if visible_area > best_area:
					best_area = visible_area
					best_quad = quad

			if not best_quad:
				# No visible quad found, use the first quad anyway
				best_quad = quads[0]
				self.logger.warning('No visible quad found, using first quad')

			# Calculate center point of the best quad
			center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
			center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

			# Ensure click point is within viewport bounds
			center_x = max(0, min(viewport_width - 1, center_x))
			center_y = max(0, min(viewport_height - 1, center_y))

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception as e:
				self.logger.debug(f'Failed to scroll element into view: {e}')

			# Perform the click using CDP
			# TODO: do occlusion detection first, if element is not on the top, fire JS-based
			# click event instead using xpath of x,y coordinate clicking, because we wont be able to click *through* occluding elements using x,y clicks
			try:
				self.logger.debug(f'👆 Dragging mouse over element before clicking x: {center_x}px y: {center_y}px ...')
				# Move mouse to element
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseMoved',
						'x': center_x,
						'y': center_y,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				# Calculate modifier bitmask for CDP
				# CDP Modifier bits: Alt=1, Control=2, Meta/Command=4, Shift=8
				modifiers = 0
				if while_holding_ctrl:
					# Use platform-appropriate modifier for "open in new tab"
					if platform.system() == 'Darwin':
						modifiers = 4  # Meta/Cmd key
						self.logger.debug('⌘ Using Cmd modifier for new tab click...')
					else:
						modifiers = 2  # Control key
						self.logger.debug('⌃ Using Ctrl modifier for new tab click...')

				# Mouse down
				self.logger.debug(f'👆🏾 Clicking x: {center_x}px y: {center_y}px with modifiers: {modifiers} ...')
				try:
					await asyncio.wait_for(
						cdp_session.cdp_client.send.Input.dispatchMouseEvent(
							params={
								'type': 'mousePressed',
								'x': center_x,
								'y': center_y,
								'button': 'left',
								'clickCount': 1,
								'modifiers': modifiers,
							},
							session_id=session_id,
						),
						timeout=1.0,  # 1 second timeout for mousePressed
					)
					await asyncio.sleep(0.08)
				except TimeoutError:
					self.logger.debug('⏱️ Mouse down timed out (likely due to dialog), continuing...')
					# Don't sleep if we timed out

				# Mouse up
				try:
					await asyncio.wait_for(
						cdp_session.cdp_client.send.Input.dispatchMouseEvent(
							params={
								'type': 'mouseReleased',
								'x': center_x,
								'y': center_y,
								'button': 'left',
								'clickCount': 1,
								'modifiers': modifiers,
							},
							session_id=session_id,
						),
						timeout=3.0,  # 1 second timeout for mouseReleased
					)
				except TimeoutError:
					self.logger.debug('⏱️ Mouse up timed out (possibly due to lag or dialog popup), continuing...')

				self.logger.debug('🖱️ Clicked successfully using x,y coordinates')
				# Return coordinates as dict for metadata
				return {'click_x': center_x, 'click_y': center_y}

			except Exception as e:
				self.logger.warning(f'CDP click failed: {type(e).__name__}: {e}')
				# Fall back to JavaScript click via CDP
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.click(); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.1)
					# Navigation is handled by BrowserSession via events
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript click also failed: {js_e}')
					raise Exception(f'Failed to click element: {e}')
			finally:
				# always re-focus back to original top-level page session context in case click opened a new tab/popup/window/dialog/etc.
				cdp_session = await self.browser_session.get_or_create_cdp_session(focus=True)
				await cdp_session.cdp_client.send.Target.activateTarget(params={'targetId': cdp_session.target_id})
				await cdp_session.cdp_client.send.Runtime.runIfWaitingForDebugger(session_id=cdp_session.session_id)

		except URLNotAllowedError as e:
			raise e
		except BrowserError as e:
			raise e
		except Exception as e:
			# Extract key element info for error message
			element_info = f'<{element_node.tag_name or "unknown"}'
			if element_node.element_index:
				element_info += f' index={element_node.element_index}'
			element_info += '>'
			raise BrowserError(
				message=f'Failed to click element: {e}',
				long_term_memory=f'Failed to click element {element_info}. The element may not be interactable or visible.',
			)

	async def _type_to_page(self, text: str):
		"""
		Type text to the page (whatever element currently has focus).
		This is used when index is 0 or when an element can't be found.
		"""
		try:
			# Get CDP client and session
			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None, focus=True)
			await cdp_session.cdp_client.send.Target.activateTarget(params={'targetId': cdp_session.target_id})

			# Type the text character by character to the focused element
			for char in text:
				# Send keydown
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyDown',
						'key': char,
					},
					session_id=cdp_session.session_id,
				)
				# Send char for actual text input
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'char',
						'text': char,
					},
					session_id=cdp_session.session_id,
				)
				# Send keyup
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyUp',
						'key': char,
					},
					session_id=cdp_session.session_id,
				)
				# Add 18ms delay between keystrokes
				await asyncio.sleep(0.018)

		except Exception as e:
			raise Exception(f'Failed to type to page: {str(e)}')

	def _get_char_modifiers_and_vk(self, char: str) -> tuple[int, int, str]:
		"""Get modifiers, virtual key code, and base key for a character.

		Returns:
			(modifiers, windowsVirtualKeyCode, base_key)
		"""
		# Characters that require Shift modifier
		shift_chars = {
			'!': ('1', 49),
			'@': ('2', 50),
			'#': ('3', 51),
			'$': ('4', 52),
			'%': ('5', 53),
			'^': ('6', 54),
			'&': ('7', 55),
			'*': ('8', 56),
			'(': ('9', 57),
			')': ('0', 48),
			'_': ('-', 189),
			'+': ('=', 187),
			'{': ('[', 219),
			'}': (']', 221),
			'|': ('\\', 220),
			':': (';', 186),
			'"': ("'", 222),
			'<': (',', 188),
			'>': ('.', 190),
			'?': ('/', 191),
			'~': ('`', 192),
		}

		# Check if character requires Shift
		if char in shift_chars:
			base_key, vk_code = shift_chars[char]
			return (8, vk_code, base_key)  # Shift=8

		# Uppercase letters require Shift
		if char.isupper():
			return (8, ord(char), char.lower())  # Shift=8

		# Lowercase letters
		if char.islower():
			return (0, ord(char.upper()), char)

		# Numbers
		if char.isdigit():
			return (0, ord(char), char)

		# Special characters without Shift
		no_shift_chars = {
			' ': 32,
			'-': 189,
			'=': 187,
			'[': 219,
			']': 221,
			'\\': 220,
			';': 186,
			"'": 222,
			',': 188,
			'.': 190,
			'/': 191,
			'`': 192,
		}

		if char in no_shift_chars:
			return (0, no_shift_chars[char], char)

		# Fallback
		return (0, ord(char.upper()) if char.isalpha() else ord(char), char)

	def _get_key_code_for_char(self, char: str) -> str:
		"""Get the proper key code for a character (like Playwright does)."""
		# Key code mapping for common characters (using proper base keys + modifiers)
		key_codes = {
			' ': 'Space',
			'.': 'Period',
			',': 'Comma',
			'-': 'Minus',
			'_': 'Minus',  # Underscore uses Minus with Shift
			'@': 'Digit2',  # @ uses Digit2 with Shift
			'!': 'Digit1',  # ! uses Digit1 with Shift (not 'Exclamation')
			'?': 'Slash',  # ? uses Slash with Shift
			':': 'Semicolon',  # : uses Semicolon with Shift
			';': 'Semicolon',
			'(': 'Digit9',  # ( uses Digit9 with Shift
			')': 'Digit0',  # ) uses Digit0 with Shift
			'[': 'BracketLeft',
			']': 'BracketRight',
			'{': 'BracketLeft',  # { uses BracketLeft with Shift
			'}': 'BracketRight',  # } uses BracketRight with Shift
			'/': 'Slash',
			'\\': 'Backslash',
			'=': 'Equal',
			'+': 'Equal',  # + uses Equal with Shift
			'*': 'Digit8',  # * uses Digit8 with Shift
			'&': 'Digit7',  # & uses Digit7 with Shift
			'%': 'Digit5',  # % uses Digit5 with Shift
			'$': 'Digit4',  # $ uses Digit4 with Shift
			'#': 'Digit3',  # # uses Digit3 with Shift
			'^': 'Digit6',  # ^ uses Digit6 with Shift
			'~': 'Backquote',  # ~ uses Backquote with Shift
			'`': 'Backquote',
			"'": 'Quote',
			'"': 'Quote',  # " uses Quote with Shift
		}

		# Numbers
		if char.isdigit():
			return f'Digit{char}'

		# Letters
		if char.isalpha():
			return f'Key{char.upper()}'

		# Special characters
		if char in key_codes:
			return key_codes[char]

		# Fallback for unknown characters
		return f'Key{char.upper()}'

	async def _clear_text_field(self, object_id: str, cdp_session) -> bool:
		"""Clear text field using multiple strategies, starting with the most reliable."""
		try:
			# Strategy 1: Direct JavaScript value setting (most reliable for modern web apps)
			self.logger.debug('🧹 Clearing text field using JavaScript value setting')

			await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': """
						function() { 
							this.value = ""; 
							this.dispatchEvent(new Event("input", { bubbles: true })); 
							this.dispatchEvent(new Event("change", { bubbles: true })); 
							return this.value;
						}
					""",
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)

			# Verify clearing worked by checking the value
			verify_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': 'function() { return this.value; }',
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)

			current_value = verify_result.get('result', {}).get('value', '')
			if not current_value:
				self.logger.debug('✅ Text field cleared successfully using JavaScript')
				return True
			else:
				self.logger.debug(f'⚠️ JavaScript clear partially failed, field still contains: "{current_value}"')

		except Exception as e:
			self.logger.debug(f'JavaScript clear failed: {e}')

		# Strategy 2: Triple-click + Delete (fallback for stubborn fields)
		try:
			self.logger.debug('🧹 Fallback: Clearing using triple-click + Delete')

			# Get element center coordinates for triple-click
			bounds_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': 'function() { return this.getBoundingClientRect(); }',
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)

			if bounds_result.get('result', {}).get('value'):
				bounds = bounds_result['result']['value']
				center_x = bounds['x'] + bounds['width'] / 2
				center_y = bounds['y'] + bounds['height'] / 2

				# Triple-click to select all text
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mousePressed',
						'x': center_x,
						'y': center_y,
						'button': 'left',
						'clickCount': 3,
					},
					session_id=cdp_session.session_id,
				)
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseReleased',
						'x': center_x,
						'y': center_y,
						'button': 'left',
						'clickCount': 3,
					},
					session_id=cdp_session.session_id,
				)

				# Delete selected text
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyDown',
						'key': 'Delete',
						'code': 'Delete',
					},
					session_id=cdp_session.session_id,
				)
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyUp',
						'key': 'Delete',
						'code': 'Delete',
					},
					session_id=cdp_session.session_id,
				)

				self.logger.debug('✅ Text field cleared using triple-click + Delete')
				return True

		except Exception as e:
			self.logger.debug(f'Triple-click clear failed: {e}')

		# Strategy 3: Keyboard shortcuts (last resort)
		try:
			import platform

			is_macos = platform.system() == 'Darwin'
			select_all_modifier = 4 if is_macos else 2  # Meta=4 (Cmd), Ctrl=2
			modifier_name = 'Cmd' if is_macos else 'Ctrl'

			self.logger.debug(f'🧹 Last resort: Clearing using {modifier_name}+A + Backspace')

			# Select all text (Ctrl/Cmd+A)
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
				params={
					'type': 'keyDown',
					'key': 'a',
					'code': 'KeyA',
					'modifiers': select_all_modifier,
				},
				session_id=cdp_session.session_id,
			)
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
				params={
					'type': 'keyUp',
					'key': 'a',
					'code': 'KeyA',
					'modifiers': select_all_modifier,
				},
				session_id=cdp_session.session_id,
			)

			# Delete selected text (Backspace)
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
				params={
					'type': 'keyDown',
					'key': 'Backspace',
					'code': 'Backspace',
				},
				session_id=cdp_session.session_id,
			)
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
				params={
					'type': 'keyUp',
					'key': 'Backspace',
					'code': 'Backspace',
				},
				session_id=cdp_session.session_id,
			)

			self.logger.debug('✅ Text field cleared using keyboard shortcuts')
			return True

		except Exception as e:
			self.logger.debug(f'All clearing strategies failed: {e}')
			return False

	async def _focus_element_simple(
		self, backend_node_id: int, object_id: str, cdp_session, input_coordinates: dict | None = None
	) -> bool:
		"""Simple focus strategy: CDP first, then click if failed."""

		# Strategy 1: Try CDP DOM.focus first
		try:
			result = await cdp_session.cdp_client.send.DOM.focus(
				params={'backendNodeId': backend_node_id},
				session_id=cdp_session.session_id,
			)
			self.logger.debug(f'Element focused using CDP DOM.focus (result: {result})')
			return True

		except Exception as e:
			self.logger.debug(f'❌ CDP DOM.focus threw exception: {type(e).__name__}: {e}')

		# Strategy 2: Try click to focus if CDP failed
		if input_coordinates and 'input_x' in input_coordinates and 'input_y' in input_coordinates:
			try:
				click_x = input_coordinates['input_x']
				click_y = input_coordinates['input_y']

				self.logger.debug(f'🎯 Attempting click-to-focus at ({click_x:.1f}, {click_y:.1f})')

				# Click to focus
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mousePressed',
						'x': click_x,
						'y': click_y,
						'button': 'left',
						'clickCount': 1,
					},
					session_id=cdp_session.session_id,
				)
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseReleased',
						'x': click_x,
						'y': click_y,
						'button': 'left',
						'clickCount': 1,
					},
					session_id=cdp_session.session_id,
				)

				self.logger.debug('✅ Element focused using click method')
				return True

			except Exception as e:
				self.logger.debug(f'Click focus failed: {e}')

		# Both strategies failed
		self.logger.warning('⚠️ All focus strategies failed')
		return False

	async def _input_text_element_node_impl(
		self, element_node: EnhancedDOMTreeNode, text: str, clear_existing: bool = True
	) -> dict | None:
		"""
		Input text into an element using pure CDP with improved focus fallbacks.
		"""

		try:
			# Get CDP client
			cdp_client = self.browser_session.cdp_client

			# Get the correct session ID for the element's iframe
			# session_id = await self._get_session_id_for_element(element_node)

			# cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=element_node.target_id, focus=True)
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get element info
			backend_node_id = element_node.backend_node_id

			# Track coordinates for metadata
			input_coordinates = None

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=cdp_session.session_id
				)
				await asyncio.sleep(0.01)
			except Exception as e:
				self.logger.warning(
					f'⚠️ Failed to focus the page {cdp_session} and scroll element {element_node} into view before typing in text: {type(e).__name__}: {e}'
				)

			# Get object ID for the element
			result = await cdp_client.send.DOM.resolveNode(
				params={'backendNodeId': backend_node_id},
				session_id=cdp_session.session_id,
			)
			assert 'object' in result and 'objectId' in result['object'], (
				'Failed to find DOM element based on backendNodeId, maybe page content changed?'
			)
			object_id = result['object']['objectId']

			# Use element_node absolute_position coordinates (correct coordinates including iframe offsets)
			if element_node.absolute_position:
				center_x = element_node.absolute_position.x + element_node.absolute_position.width / 2
				center_y = element_node.absolute_position.y + element_node.absolute_position.height / 2
				input_coordinates = {'input_x': center_x, 'input_y': center_y}
				self.logger.debug(f'Using absolute_position coordinates: x={center_x:.1f}, y={center_y:.1f}')
			else:
				input_coordinates = None
				self.logger.warning('⚠️ No absolute_position available for element')

			# Ensure we have a valid object_id before proceeding
			if not object_id:
				raise ValueError('Could not get object_id for element')

			# Step 1: Focus the element using simple strategy
			focused_successfully = await self._focus_element_simple(
				backend_node_id=backend_node_id, object_id=object_id, cdp_session=cdp_session, input_coordinates=input_coordinates
			)

			# Step 2: Clear existing text if requested
			if clear_existing and focused_successfully:
				cleared_successfully = await self._clear_text_field(object_id=object_id, cdp_session=cdp_session)
				if not cleared_successfully:
					self.logger.warning('⚠️ Text field clearing failed, typing may append to existing text')

			# Step 3: Type the text character by character using proper human-like key events
			# This emulates exactly how a human would type, which modern websites expect
			self.logger.debug(f'🎯 Typing text character by character: "{text}"')

			for i, char in enumerate(text):
				# Get proper modifiers, VK code, and base key for the character
				modifiers, vk_code, base_key = self._get_char_modifiers_and_vk(char)
				key_code = self._get_key_code_for_char(base_key)

				# self.logger.debug(f'🎯 Typing character {i + 1}/{len(text)}: "{char}" (base_key: {base_key}, code: {key_code}, modifiers: {modifiers}, vk: {vk_code})')

				# Step 1: Send keyDown event (NO text parameter)
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyDown',
						'key': base_key,
						'code': key_code,
						'modifiers': modifiers,
						'windowsVirtualKeyCode': vk_code,
					},
					session_id=cdp_session.session_id,
				)

				# Small delay to emulate human typing speed
				await asyncio.sleep(0.001)

				# Step 2: Send char event (WITH text parameter) - this is crucial for text input
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'char',
						'text': char,
						'key': char,
					},
					session_id=cdp_session.session_id,
				)

				# Step 3: Send keyUp event (NO text parameter)
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyUp',
						'key': base_key,
						'code': key_code,
						'modifiers': modifiers,
						'windowsVirtualKeyCode': vk_code,
					},
					session_id=cdp_session.session_id,
				)

				# Small delay between characters to look human (realistic typing speed)
				await asyncio.sleep(0.001)

			# Return coordinates metadata if available
			return input_coordinates

		except Exception as e:
			self.logger.error(f'Failed to input text via CDP: {type(e).__name__}: {e}')
			raise BrowserError(f'Failed to input text into element: {repr(element_node)}')

	async def _scroll_with_cdp_gesture(self, pixels: int) -> bool:
		"""
		Scroll using CDP Input.dispatchMouseEvent to simulate mouse wheel.

		Args:
			pixels: Number of pixels to scroll (positive = down, negative = up)

		Returns:
			True if successful, False if failed
		"""
		try:
			# Get CDP client and session
			assert self.browser_session.agent_focus is not None, 'CDP session not initialized - browser may not be connected yet'
			cdp_client = self.browser_session.agent_focus.cdp_client
			session_id = self.browser_session.agent_focus.session_id

			# Get viewport dimensions
			layout_metrics = await cdp_client.send.Page.getLayoutMetrics(session_id=session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Calculate center of viewport
			center_x = viewport_width / 2
			center_y = viewport_height / 2

			# For mouse wheel, positive deltaY scrolls down, negative scrolls up
			delta_y = pixels

			# Dispatch mouse wheel event
			await cdp_client.send.Input.dispatchMouseEvent(
				params={
					'type': 'mouseWheel',
					'x': center_x,
					'y': center_y,
					'deltaX': 0,
					'deltaY': delta_y,
				},
				session_id=session_id,
			)

			self.logger.debug(f'📄 Scrolled via CDP mouse wheel: {pixels}px')
			return True

		except Exception as e:
			self.logger.warning(f'❌ Scrolling via CDP failed: {type(e).__name__}: {e}')
			return False

	async def _scroll_element_container(self, element_node, pixels: int) -> bool:
		"""Try to scroll an element's container using CDP."""
		try:
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Check if this is an iframe - if so, scroll its content directly
			if element_node.tag_name and element_node.tag_name.upper() == 'IFRAME':
				# For iframes, we need to scroll the content document, not the iframe element itself
				# Use JavaScript to directly scroll the iframe's content
				backend_node_id = element_node.backend_node_id

				# Resolve the node to get an object ID
				result = await cdp_session.cdp_client.send.DOM.resolveNode(
					params={'backendNodeId': backend_node_id},
					session_id=cdp_session.session_id,
				)

				if 'object' in result and 'objectId' in result['object']:
					object_id = result['object']['objectId']

					# Scroll the iframe's content directly
					scroll_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': f"""
								function() {{
									try {{
										const doc = this.contentDocument || this.contentWindow.document;
										if (doc) {{
											const scrollElement = doc.documentElement || doc.body;
											if (scrollElement) {{
												const oldScrollTop = scrollElement.scrollTop;
												scrollElement.scrollTop += {pixels};
												const newScrollTop = scrollElement.scrollTop;
												return {{
													success: true,
													oldScrollTop: oldScrollTop,
													newScrollTop: newScrollTop,
													scrolled: newScrollTop - oldScrollTop
												}};
											}}
										}}
										return {{success: false, error: 'Could not access iframe content'}};
									}} catch (e) {{
										return {{success: false, error: e.toString()}};
									}}
								}}
							""",
							'objectId': object_id,
							'returnByValue': True,
						},
						session_id=cdp_session.session_id,
					)

					if scroll_result and 'result' in scroll_result and 'value' in scroll_result['result']:
						result_value = scroll_result['result']['value']
						if result_value.get('success'):
							self.logger.debug(f'Successfully scrolled iframe content by {result_value.get("scrolled", 0)}px')
							return True
						else:
							self.logger.debug(f'Failed to scroll iframe: {result_value.get("error", "Unknown error")}')

			# For non-iframe elements, use the standard mouse wheel approach
			# Get element bounds to know where to scroll
			backend_node_id = element_node.backend_node_id
			box_model = await cdp_session.cdp_client.send.DOM.getBoxModel(
				params={'backendNodeId': backend_node_id}, session_id=cdp_session.session_id
			)
			content_quad = box_model['model']['content']

			# Calculate center point
			center_x = (content_quad[0] + content_quad[2] + content_quad[4] + content_quad[6]) / 4
			center_y = (content_quad[1] + content_quad[3] + content_quad[5] + content_quad[7]) / 4

			# Dispatch mouse wheel event at element location
			await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
				params={
					'type': 'mouseWheel',
					'x': center_x,
					'y': center_y,
					'deltaX': 0,
					'deltaY': pixels,
				},
				session_id=cdp_session.session_id,
			)

			return True
		except Exception as e:
			self.logger.debug(f'Failed to scroll element container via CDP: {e}')
			return False

	async def _get_session_id_for_element(self, element_node: EnhancedDOMTreeNode) -> str | None:
		"""Get the appropriate CDP session ID for an element based on its frame."""
		if element_node.frame_id:
			# Element is in an iframe, need to get session for that frame
			try:
				# Get all targets
				targets = await self.browser_session.cdp_client.send.Target.getTargets()

				# Find the target for this frame
				for target in targets['targetInfos']:
					if target['type'] == 'iframe' and element_node.frame_id in str(target.get('targetId', '')):
						# Create temporary session for iframe target without switching focus
						target_id = target['targetId']
						temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)
						return temp_session.session_id

				# If frame not found in targets, use main target session
				self.logger.debug(f'Frame {element_node.frame_id} not found in targets, using main session')
			except Exception as e:
				self.logger.debug(f'Error getting frame session: {e}, using main session')

		# Use main target session
		assert self.browser_session.agent_focus is not None, 'CDP session not initialized - browser may not be connected yet'
		return self.browser_session.agent_focus.session_id

	async def on_GoBackEvent(self, event: GoBackEvent) -> None:
		"""Handle navigate back request with CDP."""
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		try:
			# Get CDP client and session

			# Get navigation history
			history = await cdp_session.cdp_client.send.Page.getNavigationHistory(session_id=cdp_session.session_id)
			current_index = history['currentIndex']
			entries = history['entries']

			# Check if we can go back
			if current_index <= 0:
				self.logger.warning('⚠️ Cannot go back - no previous entry in history')
				return

			# Navigate to the previous entry
			previous_entry_id = entries[current_index - 1]['id']
			await cdp_session.cdp_client.send.Page.navigateToHistoryEntry(
				params={'entryId': previous_entry_id}, session_id=cdp_session.session_id
			)

			# Wait for navigation
			await asyncio.sleep(0.5)
			# Navigation is handled by BrowserSession via events

			self.logger.info(f'🔙 Navigated back to {entries[current_index - 1]["url"]}')
		except Exception as e:
			raise

	async def on_GoForwardEvent(self, event: GoForwardEvent) -> None:
		"""Handle navigate forward request with CDP."""
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		try:
			# Get navigation history
			history = await cdp_session.cdp_client.send.Page.getNavigationHistory(session_id=cdp_session.session_id)
			current_index = history['currentIndex']
			entries = history['entries']

			# Check if we can go forward
			if current_index >= len(entries) - 1:
				self.logger.warning('⚠️ Cannot go forward - no next entry in history')
				return

			# Navigate to the next entry
			next_entry_id = entries[current_index + 1]['id']
			await cdp_session.cdp_client.send.Page.navigateToHistoryEntry(
				params={'entryId': next_entry_id}, session_id=cdp_session.session_id
			)

			# Wait for navigation
			await asyncio.sleep(0.5)
			# Navigation is handled by BrowserSession via events

			self.logger.info(f'🔜 Navigated forward to {entries[current_index + 1]["url"]}')
		except Exception as e:
			raise

	async def on_RefreshEvent(self, event: RefreshEvent) -> None:
		"""Handle target refresh request with CDP."""
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		try:
			# Reload the target
			await cdp_session.cdp_client.send.Page.reload(session_id=cdp_session.session_id)

			# Wait for reload
			await asyncio.sleep(1.0)

			# Note: We don't clear cached state here - let the next state fetch rebuild as needed

			# Navigation is handled by BrowserSession via events

			self.logger.info('🔄 Target refreshed')
		except Exception as e:
			raise

	async def on_WaitEvent(self, event: WaitEvent) -> None:
		"""Handle wait request."""
		try:
			# Cap wait time at maximum
			actual_seconds = min(max(event.seconds, 0), event.max_seconds)
			if actual_seconds != event.seconds:
				self.logger.info(f'🕒 Waiting for {actual_seconds} seconds (capped from {event.seconds}s)')
			else:
				self.logger.info(f'🕒 Waiting for {actual_seconds} seconds')

			await asyncio.sleep(actual_seconds)
		except Exception as e:
			raise

	async def on_SendKeysEvent(self, event: SendKeysEvent) -> None:
		"""Handle send keys request with CDP."""
		cdp_session = await self.browser_session.get_or_create_cdp_session(focus=True)
		try:
			# Parse key combination
			keys = event.keys.lower()

			# Handle special key combinations
			if '+' in keys:
				# Handle modifier keys
				parts = keys.split('+')
				key = parts[-1]

				# Calculate modifier bits inline
				# CDP Modifier bits: Alt=1, Control=2, Meta/Command=4, Shift=8
				modifiers = 0
				for part in parts[:-1]:
					part_lower = part.lower()
					if part_lower in ['alt', 'option']:
						modifiers |= 1  # Alt
					elif part_lower in ['ctrl', 'control']:
						modifiers |= 2  # Control
					elif part_lower in ['meta', 'cmd', 'command']:
						modifiers |= 4  # Meta/Command
					elif part_lower in ['shift']:
						modifiers |= 8  # Shift

				# Send key with modifiers
				# Use rawKeyDown for non-text keys (like shortcuts)
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'rawKeyDown',
						'key': key.capitalize() if len(key) == 1 else key,
						'modifiers': modifiers,
					},
					session_id=cdp_session.session_id,
				)
				await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
					params={
						'type': 'keyUp',
						'key': key.capitalize() if len(key) == 1 else key,
						'modifiers': modifiers,
					},
					session_id=cdp_session.session_id,
				)
			else:
				# Single key
				key_map = {
					'enter': 'Enter',
					'return': 'Enter',
					'tab': 'Tab',
					'delete': 'Delete',
					'backspace': 'Backspace',
					'escape': 'Escape',
					'esc': 'Escape',
					'space': ' ',
					'up': 'ArrowUp',
					'down': 'ArrowDown',
					'left': 'ArrowLeft',
					'right': 'ArrowRight',
					'pageup': 'PageUp',
					'pagedown': 'PageDown',
					'home': 'Home',
					'end': 'End',
				}

				key = key_map.get(keys, keys)

				# Keys that need 3-step sequence (produce characters)
				keys_needing_char_event = ['enter', 'return', 'space']

				# Virtual key codes for proper key identification
				virtual_key_codes = {
					'enter': 13,
					'return': 13,
					'tab': 9,
					'escape': 27,
					'esc': 27,
					'space': 32,
					'backspace': 8,
					'delete': 46,
					'up': 38,
					'down': 40,
					'left': 37,
					'right': 39,
					'home': 36,
					'end': 35,
					'pageup': 33,
					'pagedown': 34,
				}

				if keys in keys_needing_char_event:
					# 3-step sequence for keys that produce characters
					vk_code = virtual_key_codes.get(keys, 0)
					char_text = '\r' if keys in ['enter', 'return'] else ' ' if keys == 'space' else ''

					await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
						params={
							'type': 'rawKeyDown',
							'windowsVirtualKeyCode': vk_code,
							'code': key_map.get(keys, keys),
							'key': key_map.get(keys, keys),
						},
						session_id=cdp_session.session_id,
					)
					await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
						params={'type': 'char', 'text': char_text, 'unmodifiedText': char_text},
						session_id=cdp_session.session_id,
					)
					await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
						params={
							'type': 'keyUp',
							'windowsVirtualKeyCode': vk_code,
							'code': key_map.get(keys, keys),
							'key': key_map.get(keys, keys),
						},
						session_id=cdp_session.session_id,
					)
				else:
					# 2-step sequence for other keys
					key_type = 'rawKeyDown' if keys in key_map else 'keyDown'
					vk_code = virtual_key_codes.get(keys)

					if vk_code:
						# Special keys with virtual key codes
						await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
							params={
								'type': key_type,
								'key': key,
								'windowsVirtualKeyCode': vk_code,
								'code': key_map.get(keys, keys),
							},
							session_id=cdp_session.session_id,
						)
						await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
							params={
								'type': 'keyUp',
								'key': key,
								'windowsVirtualKeyCode': vk_code,
								'code': key_map.get(keys, keys),
							},
							session_id=cdp_session.session_id,
						)
					else:
						# Regular characters without virtual key codes
						await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
							params={'type': key_type, 'key': key},
							session_id=cdp_session.session_id,
						)
						await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
							params={'type': 'keyUp', 'key': key},
							session_id=cdp_session.session_id,
						)

			self.logger.info(f'⌨️ Sent keys: {event.keys}')

			# Note: We don't clear cached state on Enter; multi_act will detect DOM changes
			# and rebuild explicitly. We still wait briefly for potential navigation.
			if 'enter' in event.keys.lower() or 'return' in event.keys.lower():
				await asyncio.sleep(0.1)
		except Exception as e:
			raise

	async def on_UploadFileEvent(self, event: UploadFileEvent) -> None:
		"""Handle file upload request with CDP."""
		try:
			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Check if it's a file input
			if not self.browser_session.is_file_input(element_node):
				msg = f'Upload failed - element {index_for_logging} is not a file input.'
				raise BrowserError(message=msg, long_term_memory=msg)

			# Get CDP client and session
			cdp_client = self.browser_session.cdp_client
			session_id = await self._get_session_id_for_element(element_node)

			# Set file(s) to upload
			backend_node_id = element_node.backend_node_id
			await cdp_client.send.DOM.setFileInputFiles(
				params={
					'files': [event.file_path],
					'backendNodeId': backend_node_id,
				},
				session_id=session_id,
			)

			self.logger.info(f'📎 Uploaded file {event.file_path} to element {index_for_logging}')
		except Exception as e:
			raise

	async def on_ScrollToTextEvent(self, event: ScrollToTextEvent) -> None:
		"""Handle scroll to text request with CDP. Raises exception if text not found."""

		# TODO: handle looking for text inside cross-origin iframes as well

		# Get CDP client and session
		cdp_client = self.browser_session.cdp_client
		if self.browser_session.agent_focus is None:
			raise BrowserError('CDP session not initialized - browser may not be connected yet')
		session_id = self.browser_session.agent_focus.session_id

		# Enable DOM
		await cdp_client.send.DOM.enable(session_id=session_id)

		# Get document
		doc = await cdp_client.send.DOM.getDocument(params={'depth': -1}, session_id=session_id)
		root_node_id = doc['root']['nodeId']

		# Search for text using XPath
		search_queries = [
			f'//*[contains(text(), "{event.text}")]',
			f'//*[contains(., "{event.text}")]',
			f'//*[@*[contains(., "{event.text}")]]',
		]

		found = False
		for query in search_queries:
			try:
				# Perform search
				search_result = await cdp_client.send.DOM.performSearch(params={'query': query}, session_id=session_id)
				search_id = search_result['searchId']
				result_count = search_result['resultCount']

				if result_count > 0:
					# Get the first match
					node_ids = await cdp_client.send.DOM.getSearchResults(
						params={'searchId': search_id, 'fromIndex': 0, 'toIndex': 1},
						session_id=session_id,
					)

					if node_ids['nodeIds']:
						node_id = node_ids['nodeIds'][0]

						# Scroll the element into view
						await cdp_client.send.DOM.scrollIntoViewIfNeeded(params={'nodeId': node_id}, session_id=session_id)

						found = True
						self.logger.debug(f'📜 Scrolled to text: "{event.text}"')
						break

				# Clean up search
				await cdp_client.send.DOM.discardSearchResults(params={'searchId': search_id}, session_id=session_id)
			except Exception as e:
				self.logger.debug(f'Search query failed: {query}, error: {e}')
				continue

		if not found:
			# Fallback: Try JavaScript search
			js_result = await cdp_client.send.Runtime.evaluate(
				params={
					'expression': f'''
							(() => {{
								const walker = document.createTreeWalker(
									document.body,
									NodeFilter.SHOW_TEXT,
									null,
									false
								);
								let node;
								while (node = walker.nextNode()) {{
									if (node.textContent.includes("{event.text}")) {{
										node.parentElement.scrollIntoView({{behavior: 'smooth', block: 'center'}});
										return true;
									}}
								}}
								return false;
							}})()
						'''
				},
				session_id=session_id,
			)

		if js_result.get('result', {}).get('value'):
			self.logger.debug(f'📜 Scrolled to text: "{event.text}" (via JS)')
			return None
		else:
			self.logger.warning(f'⚠️ Text not found: "{event.text}"')
			raise BrowserError(f'Text not found: "{event.text}"', details={'text': event.text})

		# If we got here and found is True, return None (success)
		if found:
			return None
		else:
			raise BrowserError(f'Text not found: "{event.text}"', details={'text': event.text})

	async def on_GetDropdownOptionsEvent(self, event: GetDropdownOptionsEvent) -> dict[str, str]:
		"""Handle get dropdown options request with CDP."""
		try:
			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Get CDP session for this node
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Convert node to object ID for CDP operations
			try:
				object_result = await cdp_session.cdp_client.send.DOM.resolveNode(
					params={'backendNodeId': element_node.backend_node_id}, session_id=cdp_session.session_id
				)
				remote_object = object_result.get('object', {})
				object_id = remote_object.get('objectId')
				if not object_id:
					raise ValueError('Could not get object ID from resolved node')
			except Exception as e:
				raise ValueError(f'Failed to resolve node to object: {e}') from e

			# Use JavaScript to extract dropdown options
			options_script = """
			function() {
				const startElement = this;
				
				// Function to check if an element is a dropdown and extract options
				function checkDropdownElement(element) {
					// Check if it's a native select element
					if (element.tagName.toLowerCase() === 'select') {
						return {
							type: 'select',
							options: Array.from(element.options).map((opt, idx) => ({
								text: opt.text.trim(),
								value: opt.value,
								index: idx,
								selected: opt.selected
							})),
							id: element.id || '',
							name: element.name || '',
							source: 'target'
						};
					}
					
					// Check if it's an ARIA dropdown/menu
					const role = element.getAttribute('role');
					if (role === 'menu' || role === 'listbox' || role === 'combobox') {
						// Find all menu items/options
						const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
						const options = [];
						
						menuItems.forEach((item, idx) => {
							const text = item.textContent ? item.textContent.trim() : '';
							if (text) {
								options.push({
									text: text,
									value: item.getAttribute('data-value') || text,
									index: idx,
									selected: item.getAttribute('aria-selected') === 'true' || item.classList.contains('selected')
								});
							}
						});
						
						return {
							type: 'aria',
							options: options,
							id: element.id || '',
							name: element.getAttribute('aria-label') || '',
							source: 'target'
						};
					}
					
					// Check if it's a Semantic UI dropdown or similar
					if (element.classList.contains('dropdown') || element.classList.contains('ui')) {
						const menuItems = element.querySelectorAll('.item, .option, [data-value]');
						const options = [];
						
						menuItems.forEach((item, idx) => {
							const text = item.textContent ? item.textContent.trim() : '';
							if (text) {
								options.push({
									text: text,
									value: item.getAttribute('data-value') || text,
									index: idx,
									selected: item.classList.contains('selected') || item.classList.contains('active')
								});
							}
						});
						
						if (options.length > 0) {
							return {
								type: 'custom',
								options: options,
								id: element.id || '',
								name: element.getAttribute('aria-label') || '',
								source: 'target'
							};
						}
					}
					
					return null;
				}
				
				// Function to recursively search children up to specified depth
				function searchChildrenForDropdowns(element, maxDepth, currentDepth = 0) {
					if (currentDepth >= maxDepth) return null;
					
					// Check all direct children
					for (let child of element.children) {
						// Check if this child is a dropdown
						const result = checkDropdownElement(child);
						if (result) {
							result.source = `child-depth-${currentDepth + 1}`;
							return result;
						}
						
						// Recursively check this child's children
						const childResult = searchChildrenForDropdowns(child, maxDepth, currentDepth + 1);
						if (childResult) {
							return childResult;
						}
					}
					
					return null;
				}
				
				// First check the target element itself
				let dropdownResult = checkDropdownElement(startElement);
				if (dropdownResult) {
					return dropdownResult;
				}
				
				// If target element is not a dropdown, search children up to depth 4
				dropdownResult = searchChildrenForDropdowns(startElement, 4);
				if (dropdownResult) {
					return dropdownResult;
				}
				
				return {
					error: `Element and its children (depth 4) are not recognizable dropdown types (tag: ${startElement.tagName}, role: ${startElement.getAttribute('role')}, classes: ${startElement.className})`
				};
			}
			"""

			result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': options_script,
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)

			dropdown_data = result.get('result', {}).get('value', {})

			if dropdown_data.get('error'):
				raise BrowserError(message=dropdown_data['error'], long_term_memory=dropdown_data['error'])

			if not dropdown_data.get('options'):
				msg = f'No options found in dropdown at index {index_for_logging}'
				raise BrowserError(message=msg, long_term_memory=msg)

			# Format options for display
			formatted_options = []
			for opt in dropdown_data['options']:
				# Use JSON encoding to ensure exact string matching
				encoded_text = json.dumps(opt['text'])
				status = ' (selected)' if opt.get('selected') else ''
				formatted_options.append(f'{opt["index"]}: text={encoded_text}, value={json.dumps(opt["value"])}{status}')

			dropdown_type = dropdown_data.get('type', 'select')
			element_info = f'Index: {index_for_logging}, Type: {dropdown_type}, ID: {dropdown_data.get("id", "none")}, Name: {dropdown_data.get("name", "none")}'
			source_info = dropdown_data.get('source', 'unknown')

			if source_info == 'target':
				msg = f'Found {dropdown_type} dropdown ({element_info}):\n' + '\n'.join(formatted_options)
			else:
				msg = f'Found {dropdown_type} dropdown in {source_info} ({element_info}):\n' + '\n'.join(formatted_options)
			msg += f'\n\nUse the exact text or value string (without quotes) in select_dropdown_option(index={index_for_logging}, text=...)'

			if source_info == 'target':
				self.logger.info(f'📋 Found {len(dropdown_data["options"])} dropdown options for index {index_for_logging}')
			else:
				self.logger.info(
					f'📋 Found {len(dropdown_data["options"])} dropdown options for index {index_for_logging} in {source_info}'
				)

			# Create structured memory for the response
			short_term_memory = msg
			long_term_memory = f'Got dropdown options for index {index_for_logging}'

			# Return the dropdown data as a dict with structured memory
			return {
				'type': dropdown_type,
				'options': json.dumps(dropdown_data['options']),  # Convert list to JSON string for dict[str, str] type
				'element_info': element_info,
				'source': source_info,
				'formatted_options': '\n'.join(formatted_options),
				'message': msg,
				'short_term_memory': short_term_memory,
				'long_term_memory': long_term_memory,
				'element_index': str(index_for_logging),
			}

		except BrowserError:
			# Re-raise BrowserError as-is to preserve structured memory
			raise
		except TimeoutError:
			msg = f'Failed to get dropdown options for index {index_for_logging} due to timeout.'
			self.logger.error(msg)
			raise BrowserError(message=msg, long_term_memory=msg)
		except Exception as e:
			msg = f'Failed to get dropdown options for element with index {index_for_logging}'
			error_msg = f'{msg}: {str(e)}'
			self.logger.error(error_msg)
			raise BrowserError(
				message=error_msg, long_term_memory=f'Failed to get dropdown options for index {index_for_logging}.'
			)

	async def on_SelectDropdownOptionEvent(self, event: SelectDropdownOptionEvent) -> dict[str, str]:
		"""Handle select dropdown option request with CDP."""
		try:
			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'
			target_text = event.text

			# Get CDP session for this node
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Convert node to object ID for CDP operations
			try:
				object_result = await cdp_session.cdp_client.send.DOM.resolveNode(
					params={'backendNodeId': element_node.backend_node_id}, session_id=cdp_session.session_id
				)
				remote_object = object_result.get('object', {})
				object_id = remote_object.get('objectId')
				if not object_id:
					raise ValueError('Could not get object ID from resolved node')
			except Exception as e:
				raise ValueError(f'Failed to resolve node to object: {e}') from e

			try:
				# Use JavaScript to select the option
				selection_script = """
				function(targetText) {
					const startElement = this;
					
					// Function to attempt selection on a dropdown element
					function attemptSelection(element) {
						// Handle native select elements
						if (element.tagName.toLowerCase() === 'select') {
							const options = Array.from(element.options);
							const targetTextLower = targetText.toLowerCase();
							
							for (const option of options) {
								const optionTextLower = option.text.trim().toLowerCase();
								const optionValueLower = option.value.toLowerCase();
								
								// Match against both text and value (case-insensitive)
								if (optionTextLower === targetTextLower || optionValueLower === targetTextLower) {
									element.value = option.value;
									option.selected = true;
									
									// Trigger change events
									const changeEvent = new Event('change', { bubbles: true });
									element.dispatchEvent(changeEvent);
									
									return {
										success: true,
										message: `Selected option: ${option.text.trim()} (value: ${option.value})`,
										value: option.value
									};
								}
							}
							
							// Return available options as separate field
							const availableOptions = options.map(opt => ({
								text: opt.text.trim(),
								value: opt.value
							}));
							
							return {
								success: false,
								error: `Option with text or value '${targetText}' not found in select element`,
								availableOptions: availableOptions
							};
						}
						
						// Handle ARIA dropdowns/menus
						const role = element.getAttribute('role');
						if (role === 'menu' || role === 'listbox' || role === 'combobox') {
							const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
							const targetTextLower = targetText.toLowerCase();
							
							for (const item of menuItems) {
								if (item.textContent) {
									const itemTextLower = item.textContent.trim().toLowerCase();
									const itemValueLower = (item.getAttribute('data-value') || '').toLowerCase();
									
									// Match against both text and data-value (case-insensitive)
									if (itemTextLower === targetTextLower || itemValueLower === targetTextLower) {
										// Clear previous selections
										menuItems.forEach(mi => {
											mi.setAttribute('aria-selected', 'false');
											mi.classList.remove('selected');
										});
										
										// Select this item
										item.setAttribute('aria-selected', 'true');
										item.classList.add('selected');
										
										// Trigger click and change events
										item.click();
										const clickEvent = new MouseEvent('click', { view: window, bubbles: true, cancelable: true });
										item.dispatchEvent(clickEvent);
										
										return {
											success: true,
											message: `Selected ARIA menu item: ${item.textContent.trim()}`
										};
									}
								}
							}
							
							// Return available options as separate field
							const availableOptions = Array.from(menuItems).map(item => ({
								text: item.textContent ? item.textContent.trim() : '',
								value: item.getAttribute('data-value') || ''
							})).filter(opt => opt.text || opt.value);
							
							return {
								success: false,
								error: `Menu item with text or value '${targetText}' not found`,
								availableOptions: availableOptions
							};
						}
						
						// Handle Semantic UI or custom dropdowns
						if (element.classList.contains('dropdown') || element.classList.contains('ui')) {
							const menuItems = element.querySelectorAll('.item, .option, [data-value]');
							const targetTextLower = targetText.toLowerCase();
							
							for (const item of menuItems) {
								if (item.textContent) {
									const itemTextLower = item.textContent.trim().toLowerCase();
									const itemValueLower = (item.getAttribute('data-value') || '').toLowerCase();
									
									// Match against both text and data-value (case-insensitive)
									if (itemTextLower === targetTextLower || itemValueLower === targetTextLower) {
										// Clear previous selections
										menuItems.forEach(mi => {
											mi.classList.remove('selected', 'active');
										});
										
										// Select this item
										item.classList.add('selected', 'active');
										
										// Update dropdown text if there's a text element
										const textElement = element.querySelector('.text');
										if (textElement) {
											textElement.textContent = item.textContent.trim();
										}
										
										// Trigger click and change events
										item.click();
										const clickEvent = new MouseEvent('click', { view: window, bubbles: true, cancelable: true });
										item.dispatchEvent(clickEvent);
										
										// Also dispatch on the main dropdown element
										const dropdownChangeEvent = new Event('change', { bubbles: true });
										element.dispatchEvent(dropdownChangeEvent);
										
										return {
											success: true,
											message: `Selected custom dropdown item: ${item.textContent.trim()}`
										};
									}
								}
							}
							
							// Return available options as separate field
							const availableOptions = Array.from(menuItems).map(item => ({
								text: item.textContent ? item.textContent.trim() : '',
								value: item.getAttribute('data-value') || ''
							})).filter(opt => opt.text || opt.value);
							
							return {
								success: false,
								error: `Custom dropdown item with text or value '${targetText}' not found`,
								availableOptions: availableOptions
							};
						}
						
						return null; // Not a dropdown element
					}
					
					// Function to recursively search children for dropdowns
					function searchChildrenForSelection(element, maxDepth, currentDepth = 0) {
						if (currentDepth >= maxDepth) return null;
						
						// Check all direct children
						for (let child of element.children) {
							// Try selection on this child
							const result = attemptSelection(child);
							if (result && result.success) {
								return result;
							}
							
							// Recursively check this child's children
							const childResult = searchChildrenForSelection(child, maxDepth, currentDepth + 1);
							if (childResult && childResult.success) {
								return childResult;
							}
						}
						
						return null;
					}
					
					// First try the target element itself
					let selectionResult = attemptSelection(startElement);
					if (selectionResult) {
						// If attemptSelection returned a result (success or failure), use it
						// Don't search children if we found a dropdown element but selection failed
						return selectionResult;
					}
					
					// Only search children if target element is not a dropdown element
					selectionResult = searchChildrenForSelection(startElement, 4);
					if (selectionResult && selectionResult.success) {
						return selectionResult;
					}
					
					return {
						success: false,
						error: `Element and its children (depth 4) do not contain a dropdown with option '${targetText}' (tag: ${startElement.tagName}, role: ${startElement.getAttribute('role')}, classes: ${startElement.className})`
					};
				}
				"""

				result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
					params={
						'functionDeclaration': selection_script,
						'arguments': [{'value': target_text}],
						'objectId': object_id,
						'returnByValue': True,
					},
					session_id=cdp_session.session_id,
				)

				selection_result = result.get('result', {}).get('value', {})

				if selection_result.get('success'):
					msg = selection_result.get('message', f'Selected option: {target_text}')
					self.logger.debug(f'{msg}')

					# Return the result as a dict
					return {
						'success': 'true',
						'message': msg,
						'value': selection_result.get('value', target_text),
						'element_index': str(index_for_logging),
					}
				else:
					error_msg = selection_result.get('error', f'Failed to select option: {target_text}')
					available_options = selection_result.get('availableOptions', [])
					self.logger.error(f'❌ {error_msg}')
					self.logger.debug(f'Available options from JavaScript: {available_options}')

					# If we have available options, return structured error data
					if available_options:
						# Format options for short_term_memory (simple bulleted list)
						short_term_options = []
						for opt in available_options:
							if isinstance(opt, dict):
								text = opt.get('text', '').strip()
								value = opt.get('value', '').strip()
								if text:
									short_term_options.append(f'- {text}')
								elif value:
									short_term_options.append(f'- {value}')
							elif isinstance(opt, str):
								short_term_options.append(f'- {opt}')

						if short_term_options:
							short_term_memory = f'Available dropdown options at index {index_for_logging} are:\n' + '\n'.join(
								short_term_options
							)
							long_term_memory = f"Couldn't select the dropdown option at index {index_for_logging} as '{target_text}' is not one of the available options."

							# Return error result with structured memory instead of raising exception
							return {
								'success': 'false',
								'error': error_msg,
								'short_term_memory': short_term_memory,
								'long_term_memory': long_term_memory,
								'element_index': str(index_for_logging),
							}

					# Fallback to regular error result if no available options
					return {
						'success': 'false',
						'error': error_msg,
						'element_index': str(index_for_logging),
					}

			except Exception as e:
				error_msg = f'Failed to select dropdown option: {str(e)}'
				self.logger.error(error_msg)
				raise ValueError(error_msg) from e

		except Exception as e:
			error_msg = f'Failed to select dropdown option "{target_text}" for element {index_for_logging}: {str(e)}'
			self.logger.error(error_msg)
			raise ValueError(error_msg) from e

	async def on_DoubleClickEvent(self, event: DoubleClickEvent) -> dict | None:
		"""Handle double click request with CDP."""
		try:
			# Check if session is alive before attempting any operations
			if not self.browser_session.agent_focus or not self.browser_session.agent_focus.target_id:
				error_msg = 'Cannot execute double click: browser session is corrupted (target_id=None). Session may have crashed.'
				self.logger.error(f'⚠️ {error_msg}')
				raise BrowserError(error_msg)

			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'
			starting_target_id = self.browser_session.agent_focus.target_id

			# Track initial number of tabs to detect new tab opening
			initial_target_ids = await self.browser_session._cdp_get_all_pages()

			# Check if element is a file input (should not be clicked)
			if self.browser_session.is_file_input(element_node):
				msg = f'Index {index_for_logging} - has an element which opens file upload dialog. To upload files please use a specific function to upload files'
				self.logger.info(msg)
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			# Perform the actual double click using internal implementation
			click_metadata = await self._double_click_element_node_impl(element_node, button=event.button)

			# Build success message
			msg = f'Double clicked button with index {index_for_logging}: {element_node.get_all_children_text(max_depth=2)}'
			self.logger.debug(f'🖱️🖱️ {msg}')
			self.logger.debug(f'Element xpath: {element_node.xpath}')

			# Wait a bit for potential new tab to be created
			await asyncio.sleep(0.1)

			# Successfully double clicked, always reset session back to parent page session context
			self.browser_session.agent_focus = await self.browser_session.get_or_create_cdp_session(
				target_id=starting_target_id, focus=True
			)

			# Check if a new tab was opened
			after_target_ids = await self.browser_session._cdp_get_all_pages()
			new_target_ids = {t['targetId'] for t in after_target_ids} - {t['targetId'] for t in initial_target_ids}
			new_tab_opened = len(new_target_ids) > 0

			if new_target_ids:
				new_tab_msg = 'New tab opened - switching to it'
				msg += f' - {new_tab_msg}'
				self.logger.info(f'🔗 {new_tab_msg}')

				from browser_use.browser.events import SwitchTabEvent

				new_target_id = new_target_ids.pop()
				switch_event = await self.event_bus.dispatch(SwitchTabEvent(target_id=new_target_id))
				await switch_event

			# Return click metadata including new tab information
			result_metadata = click_metadata if isinstance(click_metadata, dict) else {}
			result_metadata['new_tab_opened'] = new_tab_opened

			return result_metadata
		except Exception as e:
			raise

	async def on_HoverEvent(self, event: HoverEvent) -> dict | None:
		"""Handle hover request with CDP."""
		try:
			# Check if session is alive before attempting any operations
			if not self.browser_session.agent_focus or not self.browser_session.agent_focus.target_id:
				error_msg = 'Cannot execute hover: browser session is corrupted (target_id=None). Session may have crashed.'
				self.logger.error(f'⚠️ {error_msg}')
				raise BrowserError(error_msg)

			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Perform the actual hover using internal implementation
			hover_metadata = await self._hover_element_node_impl(element_node)

			# Build success message
			msg = f'Hovered over element with index {index_for_logging}: {element_node.get_all_children_text(max_depth=2)}'
			self.logger.debug(f'👆 {msg}')
			self.logger.debug(f'Element xpath: {element_node.xpath}')

			# Return hover metadata
			return hover_metadata if isinstance(hover_metadata, dict) else {}
		except Exception as e:
			raise

	async def on_SelectTextEvent(self, event: SelectTextEvent) -> dict | None:
		"""Handle text selection request with CDP."""
		try:
			# Check if session is alive before attempting any operations
			if not self.browser_session.agent_focus or not self.browser_session.agent_focus.target_id:
				error_msg = 'Cannot execute select: browser session is corrupted (target_id=None). Session may have crashed.'
				self.logger.error(f'⚠️ {error_msg}')
				raise BrowserError(error_msg)

			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Perform the actual text selection using internal implementation
			select_metadata = await self._select_element_node_impl(
				element_node, start_offset=event.start_offset, end_offset=event.end_offset
			)

			# Build success message
			msg = f'Selected text in element with index {index_for_logging}: {element_node.get_all_children_text(max_depth=2)}'
			self.logger.debug(f'📝 {msg}')
			self.logger.debug(f'Element xpath: {element_node.xpath}')

			# Return select metadata
			return select_metadata if isinstance(select_metadata, dict) else {}
		except Exception as e:
			raise

	async def on_SubmitFormEvent(self, event: SubmitFormEvent) -> dict | None:
		"""Handle form submission request with CDP."""
		try:
			# Check if session is alive before attempting any operations
			if not self.browser_session.agent_focus or not self.browser_session.agent_focus.target_id:
				error_msg = 'Cannot execute submit: browser session is corrupted (target_id=None). Session may have crashed.'
				self.logger.error(f'⚠️ {error_msg}')
				raise BrowserError(error_msg)

			# Use the provided node
			element_node = event.node
			index_for_logging = element_node.element_index or 'unknown'

			# Perform the actual form submission using internal implementation
			submit_metadata = await self._submit_form_node_impl(element_node)

			# Build success message
			msg = f'Submitted form with index {index_for_logging}: {element_node.get_all_children_text(max_depth=2)}'
			self.logger.debug(f'📤 {msg}')
			self.logger.debug(f'Element xpath: {element_node.xpath}')

			# Return submit metadata
			return submit_metadata if isinstance(submit_metadata, dict) else {}
		except Exception as e:
			raise

	# ========== Implementation Methods for New Actions ==========

	async def _double_click_element_node_impl(self, element_node, button: str = 'left') -> dict | None:
		"""
		Double click an element using pure CDP with multiple fallback methods for getting element geometry.
		Similar to _click_element_node_impl but with double click behavior.
		"""
		try:
			# Check if element is a file input or select dropdown - these should not be clicked
			tag_name = element_node.tag_name.lower() if element_node.tag_name else ''
			element_type = element_node.attributes.get('type', '').lower() if element_node.attributes else ''

			if tag_name == 'select':
				msg = f'Cannot double click on <select> elements. Use get_dropdown_options(index={element_node.element_index}) action instead.'
				self.logger.warning(msg)
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			if tag_name == 'input' and element_type == 'file':
				msg = f'Cannot double click on file input element (index={element_node.element_index}). File uploads must be handled using upload_file_to_element action.'
				raise BrowserError(
					message=msg,
					long_term_memory=msg,
				)

			# Get CDP client
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get the correct session ID for the element's frame
			session_id = cdp_session.session_id

			# Get element bounds
			backend_node_id = element_node.backend_node_id

			# Get viewport dimensions for visibility checks
			layout_metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(session_id=session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Try multiple methods to get element geometry (same as click)
			quads = []

			# Method 1: Try DOM.getContentQuads first
			try:
				content_quads_result = await cdp_session.cdp_client.send.DOM.getContentQuads(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				if 'quads' in content_quads_result and content_quads_result['quads']:
					quads = content_quads_result['quads']
					self.logger.debug(f'Got {len(quads)} quads from DOM.getContentQuads')
			except Exception as e:
				self.logger.debug(f'DOM.getContentQuads failed: {e}')

			# Method 2: Fall back to DOM.getBoxModel
			if not quads:
				try:
					box_model = await cdp_session.cdp_client.send.DOM.getBoxModel(
						params={'backendNodeId': backend_node_id}, session_id=session_id
					)
					if 'model' in box_model and 'content' in box_model['model']:
						content_quad = box_model['model']['content']
						if len(content_quad) >= 8:
							quads = [
								[
									content_quad[0], content_quad[1],  # x1, y1
									content_quad[2], content_quad[3],  # x2, y2
									content_quad[4], content_quad[5],  # x3, y3
									content_quad[6], content_quad[7],  # x4, y4
								]
							]
							self.logger.debug('Got quad from DOM.getBoxModel')
				except Exception as e:
					self.logger.debug(f'DOM.getBoxModel failed: {e}')

			# Method 3: Fall back to JavaScript getBoundingClientRect
			if not quads:
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					if 'object' in result and 'objectId' in result['object']:
						object_id = result['object']['objectId']

						bounds_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
							params={
								'functionDeclaration': """
									function() {
										const rect = this.getBoundingClientRect();
										return {
											x: rect.left,
											y: rect.top,
											width: rect.width,
											height: rect.height
										};
									}
								""",
								'objectId': object_id,
								'returnByValue': True,
							},
							session_id=session_id,
						)

						if 'result' in bounds_result and 'value' in bounds_result['result']:
							rect = bounds_result['result']['value']
							x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
							quads = [
								[x, y, x + w, y, x + w, y + h, x, y + h]  # top-left, top-right, bottom-right, bottom-left
							]
							self.logger.debug('Got quad from getBoundingClientRect')
				except Exception as e:
					self.logger.debug(f'JavaScript getBoundingClientRect failed: {e}')

			# If we still don't have quads, fall back to JS double click
			if not quads:
				self.logger.warning('⚠️ Could not get element geometry from any method, falling back to JavaScript double click')
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.dispatchEvent(new MouseEvent("dblclick", { bubbles: true, cancelable: true })); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.05)
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript double click also failed: {js_e}')
					raise Exception(f'Failed to double click element: {js_e}')

			# Find the largest visible quad within the viewport (same logic as click)
			best_quad = None
			best_area = 0

			for quad in quads:
				if len(quad) < 8:
					continue

				# Calculate quad bounds
				xs = [quad[i] for i in range(0, 8, 2)]
				ys = [quad[i] for i in range(1, 8, 2)]
				min_x, max_x = min(xs), max(xs)
				min_y, max_y = min(ys), max(ys)

				# Check if quad intersects with viewport
				if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
					continue  # Quad is completely outside viewport

				# Calculate visible area (intersection with viewport)
				visible_min_x = max(0, min_x)
				visible_max_x = min(viewport_width, max_x)
				visible_min_y = max(0, min_y)
				visible_max_y = min(viewport_height, max_y)

				visible_width = visible_max_x - visible_min_x
				visible_height = visible_max_y - visible_min_y
				visible_area = visible_width * visible_height

				if visible_area > best_area:
					best_area = visible_area
					best_quad = quad

			if not best_quad:
				# No visible quad found, use the first quad anyway
				best_quad = quads[0]
				self.logger.warning('No visible quad found, using first quad')

			# Calculate center point of the best quad
			center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
			center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

			# Ensure click point is within viewport bounds
			center_x = max(0, min(viewport_width - 1, center_x))
			center_y = max(0, min(viewport_height - 1, center_y))

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception as e:
				self.logger.debug(f'Failed to scroll element into view: {e}')

			# Perform the double click using CDP
			try:
				self.logger.debug(f'👆👆 Double clicking element at x: {center_x}px y: {center_y}px ...')

				# Move mouse to element
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseMoved',
						'x': center_x,
						'y': center_y,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				# First click
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mousePressed',
						'x': center_x,
						'y': center_y,
						'button': button,
						'clickCount': 1,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseReleased',
						'x': center_x,
						'y': center_y,
						'button': button,
						'clickCount': 1,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				# Second click (double click)
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mousePressed',
						'x': center_x,
						'y': center_y,
						'button': button,
						'clickCount': 2,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseReleased',
						'x': center_x,
						'y': center_y,
						'button': button,
						'clickCount': 2,
					},
					session_id=session_id,
				)

				self.logger.debug('🖱️🖱️ Double clicked successfully using x,y coordinates')
				# Return coordinates as dict for metadata
				return {'click_x': center_x, 'click_y': center_y}

			except Exception as e:
				self.logger.warning(f'CDP double click failed: {type(e).__name__}: {e}')
				# Fall back to JavaScript double click via CDP
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.dispatchEvent(new MouseEvent("dblclick", { bubbles: true, cancelable: true })); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.1)
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript double click also failed: {js_e}')
					raise Exception(f'Failed to double click element: {e}')
			finally:
				# always re-focus back to original top-level page session context
				cdp_session = await self.browser_session.get_or_create_cdp_session(focus=True)
				await cdp_session.cdp_client.send.Target.activateTarget(params={'targetId': cdp_session.target_id})
				await cdp_session.cdp_client.send.Runtime.runIfWaitingForDebugger(session_id=cdp_session.session_id)

		except URLNotAllowedError as e:
			raise e
		except BrowserError as e:
			raise e
		except Exception as e:
			# Extract key element info for error message
			element_info = f'<{element_node.tag_name or "unknown"}'
			if element_node.element_index:
				element_info += f' index={element_node.element_index}'
			element_info += '>'
			raise BrowserError(
				message=f'Failed to double click element: {e}',
				long_term_memory=f'Failed to double click element {element_info}. The element may not be interactable or visible.',
			)

	async def _hover_element_node_impl(self, element_node) -> dict | None:
		"""
		Hover over an element using pure CDP.
		"""
		try:
			# Get CDP client
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get the correct session ID for the element's frame
			session_id = cdp_session.session_id

			# Get element bounds
			backend_node_id = element_node.backend_node_id

			# Get viewport dimensions for visibility checks
			layout_metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(session_id=session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Try multiple methods to get element geometry (same as click)
			quads = []

			# Method 1: Try DOM.getContentQuads first
			try:
				content_quads_result = await cdp_session.cdp_client.send.DOM.getContentQuads(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				if 'quads' in content_quads_result and content_quads_result['quads']:
					quads = content_quads_result['quads']
					self.logger.debug(f'Got {len(quads)} quads from DOM.getContentQuads')
			except Exception as e:
				self.logger.debug(f'DOM.getContentQuads failed: {e}')

			# Method 2: Fall back to DOM.getBoxModel
			if not quads:
				try:
					box_model = await cdp_session.cdp_client.send.DOM.getBoxModel(
						params={'backendNodeId': backend_node_id}, session_id=session_id
					)
					if 'model' in box_model and 'content' in box_model['model']:
						content_quad = box_model['model']['content']
						if len(content_quad) >= 8:
							quads = [
								[
									content_quad[0], content_quad[1],  # x1, y1
									content_quad[2], content_quad[3],  # x2, y2
									content_quad[4], content_quad[5],  # x3, y3
									content_quad[6], content_quad[7],  # x4, y4
								]
							]
							self.logger.debug('Got quad from DOM.getBoxModel')
				except Exception as e:
					self.logger.debug(f'DOM.getBoxModel failed: {e}')

			# Method 3: Fall back to JavaScript getBoundingClientRect
			if not quads:
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					if 'object' in result and 'objectId' in result['object']:
						object_id = result['object']['objectId']

						bounds_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
							params={
								'functionDeclaration': """
									function() {
										const rect = this.getBoundingClientRect();
										return {
											x: rect.left,
											y: rect.top,
											width: rect.width,
											height: rect.height
										};
									}
								""",
								'objectId': object_id,
								'returnByValue': True,
							},
							session_id=session_id,
						)

						if 'result' in bounds_result and 'value' in bounds_result['result']:
							rect = bounds_result['result']['value']
							x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
							quads = [
								[x, y, x + w, y, x + w, y + h, x, y + h]  # top-left, top-right, bottom-right, bottom-left
							]
							self.logger.debug('Got quad from getBoundingClientRect')
				except Exception as e:
					self.logger.debug(f'JavaScript getBoundingClientRect failed: {e}')

			# If we still don't have quads, fall back to JS hover
			if not quads:
				self.logger.warning('⚠️ Could not get element geometry from any method, falling back to JavaScript hover')
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, cancelable: true })); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.05)
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript hover also failed: {js_e}')
					raise Exception(f'Failed to hover element: {js_e}')

			# Find the largest visible quad within the viewport
			best_quad = None
			best_area = 0

			for quad in quads:
				if len(quad) < 8:
					continue

				# Calculate quad bounds
				xs = [quad[i] for i in range(0, 8, 2)]
				ys = [quad[i] for i in range(1, 8, 2)]
				min_x, max_x = min(xs), max(xs)
				min_y, max_y = min(ys), max(ys)

				# Check if quad intersects with viewport
				if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
					continue  # Quad is completely outside viewport

				# Calculate visible area (intersection with viewport)
				visible_min_x = max(0, min_x)
				visible_max_x = min(viewport_width, max_x)
				visible_min_y = max(0, min_y)
				visible_max_y = min(viewport_height, max_y)

				visible_width = visible_max_x - visible_min_x
				visible_height = visible_max_y - visible_min_y
				visible_area = visible_width * visible_height

				if visible_area > best_area:
					best_area = visible_area
					best_quad = quad

			if not best_quad:
				# No visible quad found, use the first quad anyway
				best_quad = quads[0]
				self.logger.warning('No visible quad found, using first quad')

			# Calculate center point of the best quad
			center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
			center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

			# Ensure hover point is within viewport bounds
			center_x = max(0, min(viewport_width - 1, center_x))
			center_y = max(0, min(viewport_height - 1, center_y))

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception as e:
				self.logger.debug(f'Failed to scroll element into view: {e}')

			# Perform the hover using CDP
			try:
				self.logger.debug(f'👆 Hovering over element at x: {center_x}px y: {center_y}px ...')

				# Move mouse to element
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseMoved',
						'x': center_x,
						'y': center_y,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				# Trigger mouseover event
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mousePressed',
						'x': center_x,
						'y': center_y,
						'button': 'left',
						'clickCount': 0,
					},
					session_id=session_id,
				)
				await asyncio.sleep(0.05)

				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseReleased',
						'x': center_x,
						'y': center_y,
						'button': 'left',
						'clickCount': 0,
					},
					session_id=session_id,
				)

				self.logger.debug('👆 Hovered successfully using x,y coordinates')
				# Return coordinates as dict for metadata
				return {'hover_x': center_x, 'hover_y': center_y}

			except Exception as e:
				self.logger.warning(f'CDP hover failed: {type(e).__name__}: {e}')
				# Fall back to JavaScript hover via CDP
				try:
					result = await cdp_session.cdp_client.send.DOM.resolveNode(
						params={'backendNodeId': backend_node_id},
						session_id=session_id,
					)
					assert 'object' in result and 'objectId' in result['object'], (
						'Failed to find DOM element based on backendNodeId, maybe page content changed?'
					)
					object_id = result['object']['objectId']

					await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, cancelable: true })); }',
							'objectId': object_id,
						},
						session_id=session_id,
					)
					await asyncio.sleep(0.1)
					return None
				except Exception as js_e:
					self.logger.error(f'CDP JavaScript hover also failed: {js_e}')
					raise Exception(f'Failed to hover element: {e}')
			finally:
				# always re-focus back to original top-level page session context
				cdp_session = await self.browser_session.get_or_create_cdp_session(focus=True)
				await cdp_session.cdp_client.send.Target.activateTarget(params={'targetId': cdp_session.target_id})
				await cdp_session.cdp_client.send.Runtime.runIfWaitingForDebugger(session_id=cdp_session.session_id)

		except URLNotAllowedError as e:
			raise e
		except BrowserError as e:
			raise e
		except Exception as e:
			# Extract key element info for error message
			element_info = f'<{element_node.tag_name or "unknown"}'
			if element_node.element_index:
				element_info += f' index={element_node.element_index}'
			element_info += '>'
			raise BrowserError(
				message=f'Failed to hover element: {e}',
				long_term_memory=f'Failed to hover element {element_info}. The element may not be interactable or visible.',
			)

	async def _select_element_node_impl(self, element_node, start_offset: int = 0, end_offset: int | None = None) -> dict | None:
		"""
		Select text in an element using pure CDP.
		"""
		try:
			# Get CDP client
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get the correct session ID for the element's frame
			session_id = cdp_session.session_id

			# Get element bounds
			backend_node_id = element_node.backend_node_id

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception as e:
				self.logger.debug(f'Failed to scroll element into view: {e}')

			# Get object ID for the element
			result = await cdp_session.cdp_client.send.DOM.resolveNode(
				params={'backendNodeId': backend_node_id},
				session_id=session_id,
			)
			assert 'object' in result and 'objectId' in result['object'], (
				'Failed to find DOM element based on backendNodeId, maybe page content changed?'
			)
			object_id = result['object']['objectId']

			# Use JavaScript to select text in the element
			selection_script = f"""
			function() {{
				const element = this;
				
				// Focus the element first
				element.focus();
				
				// Get the text content
				const textContent = element.textContent || element.value || '';
				
				// Determine selection range
				const start = {start_offset};
				const end = {end_offset if end_offset is not None else 'textContent.length'};
				
				// Create selection
				if (window.getSelection) {{
					const selection = window.getSelection();
					const range = document.createRange();
					
					// Try to select text within the element
					if (element.firstChild && element.firstChild.nodeType === Node.TEXT_NODE) {{
						range.setStart(element.firstChild, Math.min(start, element.firstChild.textContent.length));
						range.setEnd(element.firstChild, Math.min(end, element.firstChild.textContent.length));
					}} else {{
						// Fallback: select all text
						range.selectNodeContents(element);
					}}
					
					selection.removeAllRanges();
					selection.addRange(range);
					
					return {{
						success: true,
						selectedText: selection.toString(),
						start: start,
						end: end
					}};
				}} else {{
					return {{
						success: false,
						error: 'Selection API not supported'
					}};
				}}
			}}
			"""

			selection_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': selection_script,
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=session_id,
			)

			selection_data = selection_result.get('result', {}).get('value', {})

			if selection_data.get('success'):
				selected_text = selection_data.get('selectedText', '')
				self.logger.debug(f'Selected text: "{selected_text}"')
				return {
					'selected_text': selected_text,
					'start_offset': selection_data.get('start', start_offset),
					'end_offset': selection_data.get('end', end_offset),
				}
			else:
				error_msg = selection_data.get('error', 'Failed to select text')
				self.logger.warning(f'Text selection failed: {error_msg}')
				return {'error': error_msg}

		except Exception as e:
			# Extract key element info for error message
			element_info = f'<{element_node.tag_name or "unknown"}'
			if element_node.element_index:
				element_info += f' index={element_node.element_index}'
			element_info += '>'
			raise BrowserError(
				message=f'Failed to select text in element: {e}',
				long_term_memory=f'Failed to select text in element {element_info}. The element may not be selectable.',
			)

	async def _submit_form_node_impl(self, element_node) -> dict | None:
		"""
		Submit a form using pure CDP.
		"""
		try:
			# Get CDP client
			cdp_session = await self.browser_session.cdp_client_for_node(element_node)

			# Get the correct session ID for the element's frame
			session_id = cdp_session.session_id

			# Get element bounds
			backend_node_id = element_node.backend_node_id

			# Scroll element into view
			try:
				await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception as e:
				self.logger.debug(f'Failed to scroll element into view: {e}')

			# Get object ID for the element
			result = await cdp_session.cdp_client.send.DOM.resolveNode(
				params={'backendNodeId': backend_node_id},
				session_id=session_id,
			)
			assert 'object' in result and 'objectId' in result['object'], (
				'Failed to find DOM element based on backendNodeId, maybe page content changed?'
			)
			object_id = result['object']['objectId']

			# Use JavaScript to submit the form
			submit_script = """
			function() {
				const element = this;
				
				// Check if it's a form element
				if (element.tagName.toLowerCase() === 'form') {
					// Submit the form directly
					element.submit();
					return {
						success: true,
						method: 'form.submit()',
						action: element.action || 'current page'
					};
				}
				
				// Check if it's a submit button
				if (element.tagName.toLowerCase() === 'input' && element.type === 'submit') {
					// Find the parent form and submit it
					let form = element.closest('form');
					if (form) {
						form.submit();
						return {
							success: true,
							method: 'button.click() -> form.submit()',
							action: form.action || 'current page'
						};
					} else {
						// Just click the button
						element.click();
						return {
							success: true,
							method: 'button.click()',
							action: 'button click'
						};
					}
				}
				
				// Check if it's inside a form
				let form = element.closest('form');
				if (form) {
					form.submit();
					return {
						success: true,
						method: 'closest form.submit()',
						action: form.action || 'current page'
					};
				}
				
				return {
					success: false,
					error: 'Element is not a form or submit button and not inside a form'
				};
			}
			"""

			submit_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': submit_script,
					'objectId': object_id,
					'returnByValue': True,
				},
				session_id=session_id,
			)

			submit_data = submit_result.get('result', {}).get('value', {})

			if submit_data.get('success'):
				method = submit_data.get('method', 'unknown')
				action = submit_data.get('action', 'unknown')
				self.logger.debug(f'Form submitted using {method}, action: {action}')
				return {
					'method': method,
					'action': action,
					'success': True,
				}
			else:
				error_msg = submit_data.get('error', 'Failed to submit form')
				self.logger.warning(f'Form submission failed: {error_msg}')
				return {'error': error_msg}

		except Exception as e:
			# Extract key element info for error message
			element_info = f'<{element_node.tag_name or "unknown"}'
			if element_node.element_index:
				element_info += f' index={element_node.element_index}'
			element_info += '>'
			raise BrowserError(
				message=f'Failed to submit form: {e}',
				long_term_memory=f'Failed to submit form {element_info}. The element may not be a form or submit button.',
			)
