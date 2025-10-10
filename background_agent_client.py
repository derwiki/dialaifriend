import os
from typing import Any, Dict, Optional

import httpx


class BackgroundAgentClient:
    """Thin client for creating tasks via the Cursor Background Agent API.

    Configuration via environment variables:
    - CURSOR_BG_AGENT_CREATE_TASK_URL: Full URL to the create-task endpoint (required)
    - CURSOR_BG_AGENT_TOKEN: Bearer token for API auth (required)
    """

    def __init__(
        self,
        create_task_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.create_task_url = create_task_url or os.getenv("CURSOR_BG_AGENT_CREATE_TASK_URL")
        self.token = token or os.getenv("CURSOR_BG_AGENT_TOKEN")
        self.timeout_seconds = timeout_seconds

        if not self.create_task_url:
            raise ValueError(
                "Missing CURSOR_BG_AGENT_CREATE_TASK_URL. Set it in your environment to enable task creation."
            )
        if not self.token:
            raise ValueError(
                "Missing CURSOR_BG_AGENT_TOKEN. Set it in your environment to enable task creation."
            )

    async def create_task(
        self,
        task_text: str,
        *,
        source: str = "dial-aifriend",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a background agent task.

        The payload includes several commonly used fields to maximize compatibility
        across API versions: title, description, and prompt. The server can ignore
        unknown fields.
        """
        task_text = (task_text or "").strip()
        if not task_text:
            raise ValueError("Task text must be a non-empty string")

        title = task_text.splitlines()[0]
        if len(title) > 80:
            title = title[:77] + "..."

        payload: Dict[str, Any] = {
            "title": title,
            "description": task_text,
            "prompt": task_text,
            "source": source,
            "metadata": metadata or {},
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(self.create_task_url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"body": resp.text}
                raise RuntimeError(
                    f"Background agent task creation failed: {resp.status_code} {detail}"
                )
            return resp.json()


_singleton_client: Optional[BackgroundAgentClient] = None


def _get_client() -> BackgroundAgentClient:
    global _singleton_client
    if _singleton_client is None:
        _singleton_client = BackgroundAgentClient()
    return _singleton_client


async def create_background_agent_task(
    task_text: str, *, source: str = "dial-aifriend", metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience wrapper to create a background agent task using env configuration."""
    client = _get_client()
    return await client.create_task(task_text, source=source, metadata=metadata)
