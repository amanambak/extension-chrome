import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.models.events import AIDoneEvent
from app.models.events import AIChunkEvent
from app.models.events import ErrorEvent
from app.models.events import TranscriptEvent
from app.models.events import UtteranceCommittedEvent
from app.models.session import ConversationMessage
from app.models.session import SessionState
from app.services.deepgram_client import DeepgramClient
from app.services.gemini_client import GeminiClient
from app.services.schema_registry import get_schema_registry

logger = logging.getLogger(__name__)


class SessionRuntime:
    def __init__(self, websocket: WebSocket) -> None:
        settings = get_settings()
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())
        self.state = SessionState(session_id=self.session_id)
        self.deepgram: DeepgramClient | None = None
        self.gemini = GeminiClient()
        self.schema_registry = get_schema_registry()
        self.deepgram_task: asyncio.Task | None = None
        self.ai_lock = asyncio.Lock()
        self.closed = False
        self.connection_closed = False
        self.gemini_model_override: str | None = None
        self.finalized_segments = False
        self.finalize_task: asyncio.Task | None = None
        
        # Track activity
        self.created_at = datetime.now()
        self.last_activity_at = datetime.now()

        # Configurable parameters
        self.finalize_delay_seconds = settings.finalize_delay_seconds
        self.min_llm_interval_seconds = settings.min_llm_interval_seconds
        self.min_average_confidence = settings.min_average_confidence
        self.min_token_length = settings.min_token_length

        self.pending_incomplete_utterance = ""
        self.current_segment_confidences: list[float] = []
        self.last_llm_invoked_at = 0.0

    def update_activity(self) -> None:
        self.last_activity_at = datetime.now()

    async def run(self) -> None:
        while True:
            message = await self.websocket.receive()
            self.update_activity()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if text := message.get("text"):
                await self.handle_text_message(text)

            if data := message.get("bytes"):
                if self.deepgram is not None:
                    await self.deepgram.send_audio(data)

    async def handle_text_message(self, raw_message: str) -> None:
        self.update_activity()
        data = json.loads(raw_message)
        message_type = data.get("type")

        if message_type == "start_session":
            config = data.get("config", {})
            params = dict(config.get("deepgramParams") or {})
            params.setdefault("interim_results", "true")
            self.gemini_model_override = config.get("geminiModel")
            self.deepgram = DeepgramClient(params)
            await self.deepgram.connect()
            self.deepgram_task = asyncio.create_task(self.read_deepgram())
            await self.send_json({"type": "session_started", "sessionId": self.session_id})
            return

        if message_type == "stop_session":
            await self.close()

    async def read_deepgram(self) -> None:
        assert self.deepgram is not None
        try:
            while True:
                raw_message = await self.deepgram.recv()
                self.update_activity()
                data = json.loads(raw_message)
                await self.handle_deepgram_message(data)
        except Exception as exc:
            if not self.closed:
                logger.exception("deepgram read failed")
                await self.send_model(ErrorEvent(
                    source="Deepgram",
                    message=str(exc),
                ))

    async def handle_deepgram_message(self, data: dict) -> None:
        alternative = self._extract_primary_alternative(data)
        transcript = alternative.get("transcript", "")
        is_final = data.get("is_final", False)
        metadata = {
            "confidence": alternative.get("confidence"),
            "speech_final": data.get("speech_final", False),
        }

        if transcript:
            self._cancel_finalize_task()
            await self.send_model(TranscriptEvent(
                transcript=transcript,
                isFinal=is_final,
                metadata=metadata,
            ))

            if is_final and transcript.strip():
                confidence = metadata["confidence"]
                if self.should_capture_final_segment(transcript.strip(), confidence):
                    self.state.current_segments.append(transcript.strip())
                    self.current_segment_confidences.append(self.normalize_confidence(confidence))
                    self.finalized_segments = True
                    self._schedule_finalize()

            if metadata["speech_final"]:
                self._schedule_finalize()

        if data.get("type") == "UtteranceEnd":
            await self.send_json({"type": "utterance_end"})
            self._schedule_finalize()

    def _extract_primary_alternative(self, data: dict) -> dict:
        channel = data.get("channel", {})

        if isinstance(channel, list):
            channel = channel[0] if channel else {}

        if not isinstance(channel, dict):
            logger.warning("Unexpected Deepgram channel payload type: %s", type(channel).__name__)
            return {}

        alternatives = channel.get("alternatives", [])
        if not isinstance(alternatives, list):
            logger.warning("Unexpected Deepgram alternatives payload type: %s", type(alternatives).__name__)
            return {}

        primary = alternatives[0] if alternatives else {}
        if not isinstance(primary, dict):
            logger.warning("Unexpected Deepgram alternative item type: %s", type(primary).__name__)
            return {}

        return primary

    async def finalize_utterance(self) -> None:
        if not self.finalized_segments or not self.state.current_segments:
            return

        text = " ".join(self.state.current_segments).strip()
        self.state.current_segments = []
        self.finalized_segments = False
        average_confidence = self.get_average_confidence()
        self.current_segment_confidences = []

        if not text:
            return

        if self.pending_incomplete_utterance:
            text = f"{self.pending_incomplete_utterance} {text}".strip()
            self.pending_incomplete_utterance = ""

        if self.is_incomplete_utterance(text):
            self.pending_incomplete_utterance = text
            return

        utterance_id = f"utt-{uuid.uuid4().hex[:12]}"
        self.state.messages.append(
            ConversationMessage(type="user", text=text, utterance_id=utterance_id)
        )
        await self.send_model(UtteranceCommittedEvent(
            utteranceId=utterance_id,
            text=text,
        ))

        if self.should_extract_schema_fields(text, average_confidence):
            asyncio.create_task(self.extract_and_store_schema_fields(text))

        if self.should_invoke_llm(text, average_confidence):
            self.last_llm_invoked_at = time.monotonic()
            asyncio.create_task(self.generate_ai_response(text, utterance_id))

    def should_invoke_llm(self, utterance: str, average_confidence: float) -> bool:
        normalized = self._normalize_text(utterance)
        if not normalized:
            return False

        now = time.monotonic()
        if now - self.last_llm_invoked_at < self.min_llm_interval_seconds:
            return False

        tokens = normalized.split()
        if len(tokens) <= 2:
            return False

        greeting_phrases = {
            "hello", "helo", "hi", "hii", "hlo", "alo", "aloo", "haan", "han", "ji", "ji haan",
            "good morning", "good evening", "namaste", "namaskar", "boliye", "yes", "ok", "okay",
        }
        if normalized in greeting_phrases:
            return False

        filler_patterns = (
            "ji haan",
            "haan ji",
            "theek hai",
            "thik hai",
            "achha",
            "accha",
        )
        if any(normalized == phrase for phrase in filler_patterns):
            return False

        business_keywords = {
            "approval", "approved", "loan", "roi", "rate", "interest", "emi", "salary", "income",
            "property", "builder", "project", "sanction", "disbursement", "disburse", "fee", "fees",
            "waive", "waiver", "document", "documents", "kyc", "pan", "aadhar", "cibil", "login",
            "followup", "follow", "bank", "tenure", "eligible", "eligibility", "registration",
        }

        has_number = any(char.isdigit() for char in utterance)
        has_business_signal = has_number or any(token in business_keywords for token in tokens)

        if average_confidence < self.min_average_confidence and not has_business_signal:
            return False

        if len(tokens) < self.min_token_length and not has_business_signal:
            return False

        if len(tokens) < 10 and average_confidence < 0.82 and not has_business_signal:
            return False

        if self.looks_like_noise_or_filler(normalized):
            return False

        last_user_text = next(
            (msg.text for msg in reversed(self.state.messages[:-1]) if msg.type == "user"),
            "",
        )
        if last_user_text and self._normalize_text(last_user_text) == normalized:
            return False

        return True

    def should_extract_schema_fields(self, utterance: str, average_confidence: float) -> bool:
        normalized = self._normalize_text(utterance)
        if not normalized or average_confidence < 0.6:
            return False
        return len(normalized.split()) >= 4 or any(char.isdigit() for char in utterance)

    def should_capture_final_segment(self, transcript: str, confidence: float | None) -> bool:
        normalized = self._normalize_text(transcript)
        if not normalized:
            return False
        if len(normalized) <= 2:
            return False
        if self.looks_like_noise_or_filler(normalized):
            return False
        if confidence is not None and self.normalize_confidence(confidence) < 0.45:
            return False
        return True

    def looks_like_noise_or_filler(self, normalized: str) -> bool:
        filler_only = {
            "hmm", "hmmm", "uh", "umm", "um", "ji", "haan", "han", "hello", "helo", "hi", "ok", "okay",
            "acha", "achha", "accha", "bolo", "boliye",
        }
        tokens = normalized.split()
        if not tokens:
            return True
        if normalized in filler_only:
            return True
        unique_tokens = set(tokens)
        if len(tokens) >= 4 and len(unique_tokens) == 1:
            return True
        return False

    def normalize_confidence(self, confidence: float | None) -> float:
        if confidence is None:
            return 0.75
        return max(0.0, min(float(confidence), 1.0))

    def get_average_confidence(self) -> float:
        if not self.current_segment_confidences:
            return 0.75
        return sum(self.current_segment_confidences) / len(self.current_segment_confidences)

    def is_incomplete_utterance(self, utterance: str) -> bool:
        normalized = self._normalize_text(utterance)
        if not normalized:
            return False

        trailing_phrases = (
            "to", "toh", "ki", "aur", "or", "par", "lekin", "magar", "kyunki", "kyuki",
            "jaise", "aapne", "maine", "humne", "usme", "usmein", "isme", "ismein",
            "phir", "fir", "then", "matlab", "because",
        )
        return any(normalized.endswith(f" {phrase}") or normalized == phrase for phrase in trailing_phrases)

    async def _debounced_finalize(self) -> None:
        try:
            await asyncio.sleep(self.finalize_delay_seconds)
            await self.finalize_utterance()
        except asyncio.CancelledError:
            return

    def _schedule_finalize(self) -> None:
        self._cancel_finalize_task()
        self.finalize_task = asyncio.create_task(self._debounced_finalize())

    def _cancel_finalize_task(self) -> None:
        if self.finalize_task is not None and not self.finalize_task.done():
            self.finalize_task.cancel()
        self.finalize_task = None

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
        return normalized.strip()

    def build_recent_conversation_context(self, limit: int = 8) -> str:
        recent_messages = self.state.messages[-limit:]
        lines: list[str] = []

        known_fields_text = self.build_known_fields_text(limit=12)
        if known_fields_text:
            lines.append(f"Known customer fields: {known_fields_text}")

        for msg in recent_messages:
            speaker = "Customer" if msg.type == "user" else "Caller Assist"
            lines.append(f"{speaker}: {msg.text}")
        return "\n".join(lines) if lines else "No prior conversation context available."

    def build_known_fields_text(self, limit: int = 8) -> str:
        items = list(self.state.extracted_fields.items())
        if not items:
            return ""
        return ", ".join(f"{key}: {value}" for key, value in items[:limit])

    async def extract_and_store_schema_fields(self, utterance: str) -> None:
        conversation_context = self.build_recent_conversation_context()
        try:
            extracted = await self.gemini.extract_schema_values(
                utterance=utterance,
                conversation_context=conversation_context,
                known_fields=self.state.extracted_fields,
                schema_prompt=self.schema_registry.format_for_prompt(),
            )
        except Exception as exc:
            logger.warning("schema extraction failed: %s", exc)
            return

        for key, value in extracted.items():
            if key in self.schema_registry.fields:
                self.state.extracted_fields[key] = value

    async def generate_ai_response(self, utterance: str, utterance_id: str) -> None:
        async with self.ai_lock:
            full_text = ""
            conversation_context = self.build_recent_conversation_context()

            try:
                async for chunk in self.gemini.stream_reply(
                    utterance,
                    conversation_context,
                    self.gemini_model_override,
                ):
                    full_text += chunk
                    await self.send_model(AIChunkEvent(
                        utteranceId=utterance_id,
                        text=chunk,
                    ))
            except Exception as exc:
                await self.send_model(ErrorEvent(
                    source="Gemini",
                    message=str(exc),
                ))
                return

            full_text = self.normalize_ai_response(full_text, utterance)
            self.state.messages.append(
                ConversationMessage(
                    type="ai",
                    text=full_text,
                    utterance_id=utterance_id,
                    badge_type="suggestion",
                )
            )
            await self.send_model(AIDoneEvent(
                utteranceId=utterance_id,
                fullText=full_text,
                badgeType="suggestion",
            ))

    def normalize_ai_response(self, raw_text: str, utterance: str) -> str:
        text = re.sub(r"\s+", " ", raw_text).strip()

        summary_match = re.search(r"\[SUMMARY\](.*?)(?=\[SUGGESTION\]|$)", text, re.IGNORECASE)
        suggestion_match = re.search(r"\[SUGGESTION\](.*)$", text, re.IGNORECASE)

        summary = summary_match.group(1).strip() if summary_match else ""
        suggestion = suggestion_match.group(1).strip() if suggestion_match else ""

        summary = re.sub(r"\[/?SUMMARY\]|\[/?SUGGESTION\]", "", summary, flags=re.IGNORECASE).strip()
        suggestion = re.sub(r"\[/?SUMMARY\]|\[/?SUGGESTION\]", "", suggestion, flags=re.IGNORECASE).strip()

        if summary and summary.lower().startswith("context:"):
            summary = summary[8:].strip()
        if summary and summary.lower().startswith("topic:"):
            summary = summary[6:].strip()
        if suggestion and suggestion.lower().startswith("suggestion:"):
            suggestion = suggestion[11:].strip()
        if suggestion and suggestion.lower().startswith("topic:"):
            suggestion = suggestion[6:].strip()

        if summary and suggestion and summary in suggestion:
            suggestion = suggestion.replace(summary, "", 1).strip(" .:-")

        if not summary:
            summary = self.build_fallback_summary(utterance)
        if not suggestion:
            suggestion = "Sir/ma'am, main aapki current concern ko clear karke next step confirm kar deta hoon."

        suggestion = re.sub(r"[\u0900-\u097F]+", "", suggestion).strip()
        summary = re.sub(r"[\u0900-\u097F]+", "", summary).strip()
        summary = self.convert_summary_to_hinglish(summary)
        customer_info = self.build_known_fields_text(limit=6)

        response = f"[SUMMARY] {summary}\n"
        if customer_info:
            response += f"[CUSTOMER_INFO] {customer_info}\n"
        response += f"[SUGGESTION] {suggestion}"
        return response

    def build_fallback_summary(self, utterance: str) -> str:
        cleaned = re.sub(r"\s+", " ", utterance).strip()
        if len(cleaned) > 120:
            cleaned = f"{cleaned[:117].rstrip()}..."
        return cleaned or "Current customer discussion"

    def convert_summary_to_hinglish(self, summary: str) -> str:
        lowered = summary.lower()
        replacements = [
            ("customer is concerned about", "customer ko concern hai about"),
            ("customer confirms", "customer confirm kar raha hai"),
            ("customer is asking about", "customer pooch raha hai about"),
            ("customer is discussing", "customer discuss kar raha hai"),
            ("customer wants", "customer chah raha hai"),
            ("customer requested", "customer ne request ki hai"),
            ("customer mentioned", "customer ne mention kiya hai"),
            ("loan sanction", "loan sanction"),
            ("upfront fee", "upfront fee"),
            ("property paper check", "property paper check"),
            ("property papers", "property papers"),
            ("rate of interest", "rate of interest"),
            ("fee waiver", "fee waiver"),
            ("current status", "current status"),
            ("next action", "next action"),
            ("and is concerned about", "aur concern hai about"),
            ("and wants", "aur chah raha hai"),
        ]

        updated = summary
        for source, target in replacements:
            updated = re.sub(source, target, updated, flags=re.IGNORECASE)

        if updated == summary:
            updated = re.sub(r"^\s*customer\s+", "", updated, flags=re.IGNORECASE).strip()
            updated = re.sub(r"^\s*customer\b", "", updated, flags=re.IGNORECASE).strip(" :-")

        return updated

    async def generate_summary(self) -> dict:
        return {
            "customer_info": dict(self.state.extracted_fields),
        }

    async def send_model(self, model) -> None:
        await self.send_json(model.model_dump())

    async def send_json(self, payload: dict) -> None:
        if self.closed or self.connection_closed:
            return
        await self.websocket.send_json(payload)

    async def close(self) -> None:
        self.closed = True
        self.connection_closed = True
        self._cancel_finalize_task()

        if self.deepgram is not None:
            try:
                await self.deepgram.send_close()
            except Exception:
                pass
            await self.deepgram.close()
            self.deepgram = None

        if self.deepgram_task is not None:
            self.deepgram_task.cancel()
            self.deepgram_task = None


class SessionManager:
    def __init__(self) -> None:
        settings = get_settings()
        self._sessions: dict[str, SessionRuntime] = {}
        self.ttl = timedelta(minutes=settings.session_ttl_minutes)
        self.cleanup_interval = settings.session_cleanup_interval_seconds
        asyncio.create_task(self._cleanup_expired_sessions())

    async def _cleanup_expired_sessions(self) -> None:
        while True:
            await asyncio.sleep(self.cleanup_interval)
            now = datetime.now()
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if now - session.last_activity_at > self.ttl
            ]
            for sid in expired_ids:
                logger.info("Cleaning up expired session: %s", sid)
                await self.close_session(sid)

    async def create_session(self, websocket: WebSocket) -> SessionRuntime:
        session = SessionRuntime(websocket)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionRuntime | None:
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()
