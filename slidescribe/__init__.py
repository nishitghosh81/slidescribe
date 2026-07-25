"""SlideScribe — turn conference recordings into illustrated PDF transcripts.

Public API:
    from slidescribe import process_video, Config
    process_video("meeting.mp4", "meeting.pdf")
"""

from slidescribe.config import Config, LLMConfig
from slidescribe.pipeline import process_video, ProcessResult

__version__ = "0.1.0"
__all__ = ["process_video", "Config", "LLMConfig", "ProcessResult"]
