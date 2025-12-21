"""
Feedback endpoint for capturing user input.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models import FeedbackRequest, FeedbackResponse
from app.models.sql_feedback import Feedback
from app.database import get_db
from app.metrics import rag_feedback_total

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback for a response",
    description="""
    Submit user feedback (thumbs up/down) for a specific chat response.

    **Features:**
    - Stores feedback in PostgreSQL
    - Tracks metrics (thumbs up/down counts)
    """,
)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
):
    try:
        # Create DB record
        feedback_entry = Feedback(
            session_id=request.session_id,
            message_id=request.message_id,
            thumbs_up=request.thumbs_up,
            comment=request.comment,
        )

        # Save to DB
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)

        # Update metrics
        metric_label = "true" if request.thumbs_up else "false"
        rag_feedback_total.labels(thumbs_up=metric_label).inc()

        logger.info(
            f"Feedback received: {metric_label} for session {request.session_id}"
        )

        return FeedbackResponse(id=str(feedback_entry.id), status="received")

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback",
        )
