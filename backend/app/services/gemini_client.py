import json
import asyncio
from collections.abc import AsyncIterator

import httpx

from app.core.config import get_settings

SYSTEM_PROMPT = """You are a real-time copilot for a home-loan caller team in India.

Your output is shown to a human caller agent inside a live calling panel.
Do not sound like an AI assistant. Do not talk to the customer directly.
Help the internal caller say the next line in a natural, practical, human way.

Your job:
1. Understand the CURRENT customer utterance.
2. Use available facts from recent conversation history.
3. Produce a very short business context for the agent.
4. Produce one human-sounding Hinglish suggestion in Roman script.
5. Extract customer information based on the provided schema.

Hard rules:
- Return exactly these three sections only:
[SUMMARY] <1 short line in Roman-script Hinglish with current message context + useful customer/chat info>
[CUSTOMER_INFO] <Strict JSON object of extracted fields based on schema, or {} if nothing new found>
[SUGGESTION] <1-2 natural Hinglish lines the caller can actually speak>
- Use only Roman script. Never use Devanagari or any Hindi script characters.
- Never use [ANSWER].
- Do not say generic lines like "Kaise madad kar sakta hoon?" unless the conversation is actually starting.
- Prefer specific call-center phrasing: reassure, clarify next step, confirm details, handle objection, ask one focused follow-up.
- If the customer is only greeting, filler-speaking, or acknowledging, keep the suggestion minimal and natural.
- If there is fee, approval, sanction, eligibility, ROI, property, document, disbursement, or follow-up context, anchor the suggestion to that context.
- The summary must also be in natural Hinglish, not formal English.
- The summary must mention the current topic plus any relevant known facts already available in the recent chat.
"""

SUMMARY_PROMPT = """You are summarizing a home-loan call for an internal caller team.

Return strict JSON only in this format:
{{
  "summary": "A short paragraph in Hinglish written only in Roman script. Mention the main discussion, customer concern, current status, and next likely action."
}}

Rules:
- Use only Roman script. Never use Devanagari.
- Do not return extractedData.
- Do not return insights.
- Keep it concise and operational.
- If known customer information is provided, include those exact variable names and values inside the summary text.

Conversation:
{conversation}
"""


class GeminiClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retry_delays = (0.6, 1.2, 2.5)

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
        return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))

    async def stream_reply(
        self,
        utterance: str,
        conversation_context: str,
        known_fields: dict[str, str],
        schema_prompt: str,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        model = model_override or self.settings.gemini_model
        url = f"{self.settings.gemini_base_url}/{model}:streamGenerateContent"
        params = {
            "alt": "sse",
            "key": self.settings.gemini_api_key,
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{SYSTEM_PROMPT}\n\n"
                                f"Recent conversation history:\n{conversation_context}\n\n"
                                f"Known extracted fields so far:\n{json.dumps(known_fields, ensure_ascii=True)}\n\n"
                                f"Available schema fields:\n{schema_prompt}\n\n"
                                f"Current customer utterance:\n{utterance}"
                            )
                        }
                    ]
                }
            ]
        }

        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                    async with client.stream("POST", url, params=params, json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = json.loads(line[6:])
                            text = (
                                data.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if text:
                                yield text
                return
            except Exception as exc:
                if attempt >= len(self.retry_delays) or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self.retry_delays[attempt])

    async def generate_summary(self, conversation: str) -> dict:
        url = f"{self.settings.gemini_base_url}/{self.settings.summary_model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": SUMMARY_PROMPT.format(conversation=conversation)
                        }
                    ]
                }
            ]
        }

        data = await self._post_json_with_retry(url, params, payload)

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {
                "summary": "Summary response could not be parsed.",
            }

        return json.loads(text[start : end + 1])

    async def extract_schema_values(
        self,
        utterance: str,
        conversation_context: str,
        known_fields: dict[str, str],
        schema_prompt: str,
    ) -> dict[str, str]:
        prompt = f"""You extract customer information from a home-loan call.

Return strict JSON only.
Use only exact variable names from the provided schema.
Extract only fields that are explicitly mentioned or strongly implied from the current utterance with nearby context.
Do not invent values.
If nothing is found, return {{}}.

Known extracted fields so far:
{json.dumps(known_fields, ensure_ascii=True)}

Recent conversation:
{conversation_context}

Current utterance:
{utterance}

Available schema fields:
{schema_prompt}
"""

        url = f"{self.settings.gemini_base_url}/{self.settings.summary_model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        data = await self._post_json_with_retry(url, params, payload)

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}

        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            return {}

        return {
            str(key): str(value)
            for key, value in parsed.items()
            if value is not None and str(value).strip()
        }

    async def _post_json_with_retry(self, url: str, params: dict, payload: dict) -> dict:
        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                    response = await client.post(url, params=params, json=payload)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:
                if attempt >= len(self.retry_delays) or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self.retry_delays[attempt])

        raise RuntimeError("unreachable")
