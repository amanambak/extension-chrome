from pydantic import BaseModel, Field, ConfigDict
from fastapi import APIRouter, HTTPException
from fastapi import WebSocket, WebSocketDisconnect

from app.services.gemini_client import GeminiClient
from app.services.session_manager import SessionManager
from app.services.schema_registry import get_schema_registry

router = APIRouter()
session_manager = SessionManager()
gemini_client = GeminiClient()
schema_registry = get_schema_registry()


class SummaryRequest(BaseModel):
    conversation: str = Field(..., max_length=50000)

    model_config = ConfigDict(
        max_str_length=50000
    )


@router.websocket("/ws/session")
async def session_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = await session_manager.create_session(websocket)

    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    finally:
        await session_manager.close_session(session.session_id)


@router.get("/api/sessions/{session_id}/summary")
async def session_summary(session_id: str) -> dict:
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return await session.generate_summary()


@router.post("/api/summary")
async def ad_hoc_summary(request: SummaryRequest) -> dict:
    customer_info = await gemini_client.extract_schema_values(
        utterance=request.conversation,
        conversation_context=request.conversation,
        known_fields={},
        schema_prompt=schema_registry.format_for_prompt(),
    )
    filtered = {
        key: value for key, value in customer_info.items()
        if key in schema_registry.fields
    }
    return {"customer_info": filtered}
