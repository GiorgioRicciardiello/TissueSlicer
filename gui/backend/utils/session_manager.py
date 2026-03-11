"""
Session management for tissue selection GUI.

Stores image loaders, scale factors, and user selections per session.
Auto-cleanup of idle sessions after configurable timeout.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manage user sessions for the tissue selection GUI.

    Each session stores:
    - image_path: Path to the loaded OME-TIFF
    - loader: OmeTiffImageLoader instance (lazy-loaded image)
    - scale_y, scale_x: Downsampled → full-resolution scale factors
    - selections: List of polygon selections with metadata
    - created_at: Session creation timestamp
    """

    def __init__(self, auto_cleanup_minutes: int = 60):
        """
        Initialize session manager.

        Args:
            auto_cleanup_minutes: Auto-delete sessions idle for this many minutes
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.auto_cleanup_minutes = auto_cleanup_minutes
        logger.info(f"SessionManager initialized (cleanup after {auto_cleanup_minutes} min)")

    def create_session(
        self,
        image_path: str,
        loader: Any,
        scale_y: float,
        scale_x: float
    ) -> str:
        """
        Create a new session for this image.

        Args:
            image_path: Path to OME-TIFF file
            loader: OmeTiffImageLoader instance
            scale_y, scale_x: Downsampled → full-resolution scale factors

        Returns:
            Session ID (timestamp string, ~16 chars)
        """
        # Use timestamp as unique session ID
        timestamp = datetime.now().isoformat()

        session = {
            'image_path': image_path,
            'loader': loader,
            'scale_y': scale_y,
            'scale_x': scale_x,
            'selections': [],
            'calibrations': {},  # Channel calibrations: {ch_idx: {p2, p98, min, max, mean}}
            'created_at': timestamp,
            'last_accessed': timestamp
        }

        self.sessions[timestamp] = session
        logger.info(
            f"Session created: {timestamp} for image {image_path} "
            f"(scale=({scale_y:.2f}×, {scale_x:.2f}×))"
        )

        return timestamp

    def get_session(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.

        Args:
            timestamp: Session ID returned by create_session()

        Returns:
            Session dict, or None if not found or expired
        """
        if timestamp not in self.sessions:
            logger.warning(f"Session not found: {timestamp}")
            return None

        session = self.sessions[timestamp]

        # Update last-accessed time (for auto-cleanup)
        session['last_accessed'] = datetime.now().isoformat()

        return session

    def add_selection(self, timestamp: str, selection: Dict[str, Any]) -> None:
        """
        Add a polygon selection to a session.

        Args:
            timestamp: Session ID
            selection: Dict with keys: id, name, polygon_coords, padding_px, force_square, created_at
        """
        session = self.get_session(timestamp)
        if not session:
            raise ValueError(f"Session not found: {timestamp}")

        session['selections'].append(selection)
        logger.debug(
            f"Selection added to session {timestamp}: "
            f"name='{selection['name']}', id={selection['id']}"
        )

    def remove_selection(self, timestamp: str, selection_id: str) -> None:
        """
        Remove a selection from a session.

        Args:
            timestamp: Session ID
            selection_id: Selection ID to remove
        """
        session = self.get_session(timestamp)
        if not session:
            raise ValueError(f"Session not found: {timestamp}")

        session['selections'] = [
            s for s in session['selections']
            if s['id'] != selection_id
        ]
        logger.debug(f"Selection removed: {selection_id} from session {timestamp}")

    def clear_session(self, timestamp: str) -> None:
        """
        Clear all selections in a session.

        Args:
            timestamp: Session ID
        """
        session = self.get_session(timestamp)
        if not session:
            raise ValueError(f"Session not found: {timestamp}")

        session['selections'] = []
        logger.info(f"Session cleared: {timestamp}")

    def set_channel_calibration(self, timestamp: str, channel_idx: int, calibration: Dict[str, Any]) -> None:
        """
        Store calibration for a channel in the session.

        Args:
            timestamp: Session ID
            channel_idx: Channel index
            calibration: Dict with keys p2, p98, min, max, mean
        """
        session = self.get_session(timestamp)
        if not session:
            raise ValueError(f"Session not found: {timestamp}")

        session['calibrations'][channel_idx] = calibration
        logger.debug(f"Calibration set for channel {channel_idx} in session {timestamp}")

    def get_channel_calibration(self, timestamp: str, channel_idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve calibration for a channel from the session.

        Args:
            timestamp: Session ID
            channel_idx: Channel index

        Returns:
            Calibration dict, or None if not found
        """
        session = self.get_session(timestamp)
        if not session:
            return None

        return session.get('calibrations', {}).get(channel_idx)

    def cleanup_old_sessions(self) -> None:
        """
        Remove sessions idle for longer than auto_cleanup_minutes.

        Called automatically on each request by Flask @before_request hook.
        """
        now = datetime.fromisoformat(datetime.now().isoformat())
        cutoff = now - timedelta(minutes=self.auto_cleanup_minutes)

        expired_sessions = []
        for timestamp, session in list(self.sessions.items()):
            last_accessed_str = session.get('last_accessed', session['created_at'])
            last_accessed = datetime.fromisoformat(last_accessed_str)

            if last_accessed < cutoff:
                expired_sessions.append(timestamp)

        for timestamp in expired_sessions:
            # Close loader if it has a close method
            loader = self.sessions[timestamp].get('loader')
            if loader and hasattr(loader, 'close'):
                try:
                    loader.close()
                except Exception as e:
                    logger.warning(f"Error closing loader for {timestamp}: {e}")

            del self.sessions[timestamp]
            logger.info(f"Session expired and cleaned up: {timestamp}")

        if expired_sessions:
            logger.debug(
                f"Cleanup completed: {len(expired_sessions)} sessions removed, "
                f"{len(self.sessions)} sessions remain"
            )
