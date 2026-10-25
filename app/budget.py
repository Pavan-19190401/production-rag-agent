import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.config import settings


class BudgetExceededError(Exception):
    """Raised when the daily LLM spend limit has been exceeded."""
    pass


class BudgetManager:
    """Tracks daily token usage and estimated spend against configured caps."""

    def __init__(self, daily_budget_usd: float = settings.DAILY_BUDGET_USD):
        self.daily_budget_usd = daily_budget_usd
        self.current_date = self._get_today()
        self.daily_spend_usd = 0.0
        self.daily_tokens = 0
        self.total_requests = 0
        self.lock = threading.Lock()

    def _get_today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_and_reset_day(self) -> None:
        today = self._get_today()
        if today != self.current_date:
            self.current_date = today
            self.daily_spend_usd = 0.0
            self.daily_tokens = 0
            self.total_requests = 0

    def check_budget(self) -> None:
        """Verifies that the daily spend cap has not been breached."""
        with self.lock:
            self._check_and_reset_day()
            if self.daily_spend_usd >= self.daily_budget_usd:
                raise BudgetExceededError(
                    f"Daily LLM budget cap of ${self.daily_budget_usd:.2f} has been reached "
                    f"(Current spend: ${self.daily_spend_usd:.4f}). Requests are temporarily throttled."
                )

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: Optional[float] = None
    ) -> None:
        """Records tokens and estimated cost for an executed query."""
        with self.lock:
            self._check_and_reset_day()
            tokens = prompt_tokens + completion_tokens
            self.daily_tokens += tokens
            self.total_requests += 1

            if cost_usd is not None:
                self.daily_spend_usd += cost_usd
            else:
                # Approximate blend for modern cost-effective LLMs (~$0.30 per 1M tokens)
                estimated_cost = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)
                self.daily_spend_usd += estimated_cost

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            self._check_and_reset_day()
            return {
                "date": self.current_date,
                "daily_spend_usd": round(self.daily_spend_usd, 4),
                "daily_budget_usd": self.daily_budget_usd,
                "daily_tokens": self.daily_tokens,
                "total_requests": self.total_requests,
                "budget_remaining_usd": max(0.0, round(self.daily_budget_usd - self.daily_spend_usd, 4))
            }


# Global budget manager
budget_manager = BudgetManager()
