from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# Action Input Models
class SearchGoogleAction(BaseModel):
	query: str


class GoToUrlAction(BaseModel):
	url: str
	new_tab: bool = False  # True to open in new tab, False to navigate in current tab


class ClickElementAction(BaseModel):
	index: int = Field(ge=1, description='index of the element to click')
	while_holding_ctrl: bool | None = Field(
		default=None,
		description='Set to True to open the navigation in a new background tab (Ctrl+Click behavior). Optional.',
	)
	# expect_download: bool = Field(default=False, description='set True if expecting a download, False otherwise')  # moved to downloads_watchdog.py
	# click_count: int = 1  # TODO


class InputTextAction(BaseModel):
	index: int = Field(ge=0, description='index of the element to input text into, 0 is the page')
	text: str
	clear_existing: bool = Field(default=True, description='set True to clear existing text, False to append to existing text')


class DoneAction(BaseModel):
	text: str
	success: bool
	files_to_display: list[str] | None = []


T = TypeVar('T', bound=BaseModel)


class StructuredOutputAction(BaseModel, Generic[T]):
	success: bool = True
	data: T


class SwitchTabAction(BaseModel):
	tab_id: str = Field(
		min_length=4,
		max_length=4,
		description='Last 4 chars of TargetID',
	)  # last 4 chars of TargetID


class CloseTabAction(BaseModel):
	tab_id: str = Field(min_length=4, max_length=4, description='4 character Tab ID')  # last 4 chars of TargetID


class ScrollAction(BaseModel):
	down: bool | None = True # True to scroll down, False to scroll up
	num_pages: float  # Number of pages to scroll (0.5 = half page, 1.0 = one page, etc.)
	frame_element_index: int | None = None  # Optional element index to find scroll container for


class SendKeysAction(BaseModel):
	keys: str


class UploadFileAction(BaseModel):
	index: int
	path: str


class ExtractPageContentAction(BaseModel):
	value: str


class NoParamsAction(BaseModel):
	"""
	Accepts absolutely anything in the incoming data
	and discards it, so the final parsed model is empty.
	"""

	model_config = ConfigDict(extra='ignore')
	# No fields defined - all inputs are ignored automatically


class GetDropdownOptionsAction(BaseModel):
	index: int = Field(ge=1, description='index of the dropdown element to get the option values for')


class SelectDropdownOptionAction(BaseModel):
	index: int = Field(ge=1, description='index of the dropdown element to select an option for')
	text: str = Field(description='the text or exact value of the option to select')


class DoubleClickElementAction(BaseModel):
	index: int = Field(ge=1, description='index of the element to double click')
	button: Literal['left', 'right', 'middle'] = Field(default='left', description='mouse button to use for double click')


class HoverElementAction(BaseModel):
	index: int = Field(ge=1, description='index of the element to hover over')


class SelectElementAction(BaseModel):
	index: int = Field(ge=1, description='index of the element to select text in')
	start_offset: int = Field(default=0, description='start position for text selection')
	end_offset: int | None = Field(default=None, description='end position for text selection (None means select all)')


class SubmitFormAction(BaseModel):
	index: int = Field(ge=1, description='index of the form element to submit')