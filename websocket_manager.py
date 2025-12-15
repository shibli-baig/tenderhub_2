"""
WebSocket Connection Manager for Real-time Updates
Handles WebSocket connections for tender workspaces, enabling real-time chat and task updates.
"""

from typing import Dict, Set, Optional
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for tender workspaces.
    Uses room-based architecture where each tender_id represents a room.
    """

    def __init__(self):
        # Dictionary mapping tender_id to set of active WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Track user/employee info for each connection
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, tender_id: str, user_type: str, user_id: str, user_name: str, already_accepted: bool = False):
        """
        Accept a new WebSocket connection and add it to the tender's room.

        Args:
            websocket: The WebSocket connection
            tender_id: The tender ID (room identifier)
            user_type: 'manager' or 'employee'
            user_id: User or Employee ID
            user_name: Display name
            already_accepted: If True, skip accepting the websocket (it's already been accepted)
        """
        if not already_accepted:
            await websocket.accept()

        # Create room if it doesn't exist
        if tender_id not in self.active_connections:
            self.active_connections[tender_id] = set()

        # Add connection to room
        self.active_connections[tender_id].add(websocket)

        # Store connection metadata
        self.connection_info[websocket] = {
            "tender_id": tender_id,
            "user_type": user_type,
            "user_id": user_id,
            "user_name": user_name
        }

        logger.info(f"WebSocket connected: {user_type} {user_name} ({user_id}) to tender {tender_id}")

        # Notify others in the room
        await self.broadcast_to_room(
            tender_id,
            {
                "type": "user_connected",
                "user_type": user_type,
                "user_name": user_name
            },
            exclude=websocket
        )

    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from its room.

        Args:
            websocket: The WebSocket connection to remove
        """
        if websocket in self.connection_info:
            info = self.connection_info[websocket]
            tender_id = info["tender_id"]

            # Remove from room
            if tender_id in self.active_connections:
                self.active_connections[tender_id].discard(websocket)

                # Clean up empty rooms
                if not self.active_connections[tender_id]:
                    del self.active_connections[tender_id]

            logger.info(f"WebSocket disconnected: {info['user_type']} {info['user_name']} from tender {tender_id}")

            # Remove metadata
            del self.connection_info[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: The message dictionary to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_room(self, tender_id: str, message: dict, exclude: Optional[WebSocket] = None):
        """
        Broadcast a message to all connections in a tender's room.

        Args:
            tender_id: The tender ID (room identifier)
            message: The message dictionary to broadcast
            exclude: Optional WebSocket to exclude from broadcast (e.g., sender)
        """
        if tender_id not in self.active_connections:
            return

        # Get all connections in this room
        connections = self.active_connections[tender_id].copy()

        # Remove failed connections
        failed_connections = set()

        for connection in connections:
            if exclude and connection == exclude:
                continue

            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                failed_connections.add(connection)

        # Clean up failed connections
        for connection in failed_connections:
            self.disconnect(connection)

    async def broadcast_task_event(self, tender_id: str, event_type: str, data: dict):
        """
        Broadcast a task-related event to all connections in a room.

        Args:
            tender_id: The tender ID
            event_type: Type of event (task_created, task_updated, etc.)
            data: Event data
        """
        message = {
            "type": event_type,
            "data": data
        }
        await self.broadcast_to_room(tender_id, message)

    async def broadcast_chat_message(self, tender_id: str, message_data: dict):
        """
        Broadcast a chat message to all connections in a room.

        Args:
            tender_id: The tender ID
            message_data: Message data including sender info, text, timestamp
        """
        message = {
            "type": "chat_message",
            "data": message_data
        }
        await self.broadcast_to_room(tender_id, message)

    def get_room_size(self, tender_id: str) -> int:
        """
        Get the number of active connections in a room.

        Args:
            tender_id: The tender ID

        Returns:
            Number of active connections
        """
        if tender_id in self.active_connections:
            return len(self.active_connections[tender_id])
        return 0


# Global connection manager instance
manager = ConnectionManager()
