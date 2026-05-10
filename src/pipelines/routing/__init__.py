"""Sprint 8: семантическое ветвление запросов в Telegram-боте (config: semantic_routing)."""

from src.pipelines.routing.router import (
    RoutingDecision,
    SemanticRoutingPort,
    create_semantic_routing_service,
)

__all__ = [
    "RoutingDecision",
    "SemanticRoutingPort",
    "create_semantic_routing_service",
]
