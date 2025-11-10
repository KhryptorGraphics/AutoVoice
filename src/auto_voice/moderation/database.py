"""
Database logging for content moderation audit trails.

Provides persistent storage of moderation decisions, user actions, and system events.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
import json

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available. Database logging will be disabled.")

logger = logging.getLogger(__name__)

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class ModerationLog(Base):
        """Moderation event log table."""
        __tablename__ = 'moderation_logs'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
        event_type = Column(String(50), nullable=False)  # 'nsfw_check', 'age_verify', 'deepfake_check', etc.
        content_type = Column(String(50))  # 'image', 'audio', 'video', 'text'
        content_id = Column(String(255))  # Unique identifier for content
        user_id = Column(String(255))  # User identifier
        
        # Detection results
        is_flagged = Column(Boolean, default=False)
        confidence_score = Column(Float)
        threshold = Column(Float)
        
        # Metadata
        detector_model = Column(String(100))  # Model/service used
        processing_time_ms = Column(Float)
        metadata = Column(JSON)  # Additional context
        
        # Action taken
        action = Column(String(50))  # 'blocked', 'allowed', 'review_required'
        notes = Column(Text)
    
    class UserAction(Base):
        """User action audit log."""
        __tablename__ = 'user_actions'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
        user_id = Column(String(255), nullable=False)
        action_type = Column(String(50), nullable=False)  # 'upload', 'generate', 'download', etc.
        resource_type = Column(String(50))
        resource_id = Column(String(255))
        ip_address = Column(String(45))  # IPv6 compatible
        user_agent = Column(Text)
        metadata = Column(JSON)


class ModerationDatabase:
    """Database manager for moderation logging and audit trails."""
    
    def __init__(self, database_url: str = "sqlite:///moderation.db"):
        """
        Initialize moderation database.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.available = SQLALCHEMY_AVAILABLE
        
        if not self.available:
            logger.warning("SQLAlchemy not available. Database logging disabled.")
            return
        
        try:
            self.engine = create_engine(database_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Moderation database initialized: {database_url}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.available = False
    
    def log_moderation_event(
        self,
        event_type: str,
        content_type: str,
        is_flagged: bool,
        confidence_score: float,
        threshold: float,
        detector_model: str,
        content_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: str = "allowed",
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Log a moderation event to the database.
        
        Args:
            event_type: Type of moderation check
            content_type: Type of content checked
            is_flagged: Whether content was flagged
            confidence_score: Detection confidence
            threshold: Threshold used
            detector_model: Model/service name
            content_id: Content identifier
            user_id: User identifier
            action: Action taken
            processing_time_ms: Processing time
            metadata: Additional metadata
            notes: Optional notes
            
        Returns:
            True if logged successfully
        """
        if not self.available:
            return False
        
        try:
            session = self.SessionLocal()
            
            log_entry = ModerationLog(
                event_type=event_type,
                content_type=content_type,
                content_id=content_id,
                user_id=user_id,
                is_flagged=is_flagged,
                confidence_score=confidence_score,
                threshold=threshold,
                detector_model=detector_model,
                processing_time_ms=processing_time_ms,
                metadata=metadata,
                action=action,
                notes=notes
            )
            
            session.add(log_entry)
            session.commit()
            session.close()
            
            logger.debug(f"Logged moderation event: {event_type} - {action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log moderation event: {e}")
            return False
    
    def log_user_action(
        self,
        user_id: str,
        action_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Log a user action."""
        if not self.available:
            return False
        
        try:
            session = self.SessionLocal()
            
            action = UserAction(
                user_id=user_id,
                action_type=action_type,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata
            )
            
            session.add(action)
            session.commit()
            session.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
            return False
    
    def get_moderation_stats(self, days: int = 7) -> Dict:
        """Get moderation statistics for the last N days."""
        if not self.available:
            return {}
        
        try:
            session = self.SessionLocal()
            
            # Calculate date threshold
            from datetime import timedelta
            threshold_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Query logs
            logs = session.query(ModerationLog).filter(
                ModerationLog.timestamp >= threshold_date
            ).all()
            
            total = len(logs)
            flagged = sum(1 for log in logs if log.is_flagged)
            
            stats = {
                'total_checks': total,
                'flagged_count': flagged,
                'allowed_count': total - flagged,
                'flagged_percentage': (flagged / total * 100) if total > 0 else 0,
                'period_days': days
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get moderation stats: {e}")
            return {}

