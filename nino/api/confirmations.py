"""REST API endpoints for interactive tool confirmation management."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from nino.auth import verify_token
from nino.tools.confirmation import Confirmation, ConfirmationManager

router = APIRouter(prefix="/confirmations", tags=["confirmations"])


def get_confirmation_mgr(request: Request) -> ConfirmationManager:
    """Retrieve ConfirmationManager from application state."""
    mgr = getattr(request.app.state, "confirmation_manager", None)
    if not mgr:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ConfirmationManager is not initialized on application state.",
        )
    return mgr


class ConfirmationStatusResponse(BaseModel):
    status: str
    confirmation_id: str
    message: str


@router.get("", response_model=list[Confirmation], summary="List pending confirmations", dependencies=[Depends(verify_token)])
async def list_pending_confirmations(
    mgr: ConfirmationManager = Depends(get_confirmation_mgr),
):
    """Retrieve all currently pending, unexpired tool confirmations."""
    return mgr.list_pending()


@router.post("/{confirmation_id}/approve", response_model=ConfirmationStatusResponse, summary="Approve confirmation", dependencies=[Depends(verify_token)])
async def approve_confirmation(
    confirmation_id: str,
    mgr: ConfirmationManager = Depends(get_confirmation_mgr),
):
    """Approve a pending confirmation token."""
    success, msg = mgr.approve(confirmation_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    return ConfirmationStatusResponse(
        status="approved",
        confirmation_id=confirmation_id,
        message=msg,
    )


@router.post("/{confirmation_id}/deny", response_model=ConfirmationStatusResponse, summary="Deny confirmation", dependencies=[Depends(verify_token)])
async def deny_confirmation(
    confirmation_id: str,
    mgr: ConfirmationManager = Depends(get_confirmation_mgr),
):
    """Deny a pending confirmation token."""
    success, msg = mgr.deny(confirmation_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    return ConfirmationStatusResponse(
        status="denied",
        confirmation_id=confirmation_id,
        message=msg,
    )
