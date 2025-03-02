from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
from app.logger_config import logger

@dataclass
class Message:
    """Single message in conversation"""
    timestamp: datetime
    query: str
    response: str
    pages_used: List[int]
    collection_name: str

@dataclass
class Conversation:
    """Conversation history for a session"""
    session_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    collection_name: str  # Current PDF being discussed
    
    def add_message(self, query: str, response: str, pages_used: List[int]) -> None:
        """Add new message to conversation"""
        self.messages.append(Message(
            timestamp=datetime.now(),
            query=query,
            response=response,
            pages_used=pages_used,
            collection_name=self.collection_name
        ))
        self.updated_at = datetime.now()
        
    def get_recent_context(self, max_messages: int = 3) -> List[Message]:
        """Get recent messages for context"""
        return self.messages[-max_messages:] if self.messages else []

class ConversationService:
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        
    def get_or_create_conversation(self, session_id: str, collection_name: str) -> Conversation:
        """Get existing conversation or create new one"""
        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(
                session_id=session_id,
                messages=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                collection_name=collection_name
            )
        return self._conversations[session_id]
        
    def add_message(self, session_id: str, query: str, response: str, pages_used: List[int], collection_name: str) -> None:
        """Add message to conversation"""
        conv = self.get_or_create_conversation(session_id, collection_name)
        conv.add_message(query, response, pages_used)
        logger.info(f"Added message to conversation {session_id}")
        
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get conversation history"""
        if session_id in self._conversations:
            messages = self._conversations[session_id].messages
            return messages[-limit:] if len(messages) > limit else messages
        return []
        
    def get_recent_context(self, session_id: str, max_messages: int = 3) -> List[Message]:
        """Get recent conversation context"""
        if session_id in self._conversations:
            return self._conversations[session_id].get_recent_context(max_messages)
        return []