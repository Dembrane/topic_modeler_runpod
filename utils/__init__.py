# Backward compatibility imports - main entry points
from services.view_processor import get_views_aspects, get_views_aspects_fallback

# Export the main functions that are used by handler.py
__all__ = ["get_views_aspects", "get_views_aspects_fallback"]
