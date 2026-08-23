from __future__ import annotations

from fastapi import FastAPI

from agent.conversation_styles import ConversationStyleService

from .attachments import DashboardAttachmentStore
from .broker import DashboardChatBroker
from .contracts import DashboardRuntimeServices
from .document_parser import DashboardDocumentParser
from .routes.attention import register_attention_routes
from .routes.chat import register_chat_routes
from .routes.memory import register_memory_routes
from .routes.models import register_model_routes
from .routes.personal import register_personal_routes
from .routes.rhythm import register_rhythm_routes
from .routes.sources import register_source_routes
from .routes.system import register_system_routes
from .routes.workflows import register_workflow_routes


def register_dashboard_management(
    app: FastAPI,
    services: DashboardRuntimeServices | None,
) -> None:
    """Attach dashboard management routes backed by one shared runtime."""

    if services is None:
        return

    if services.conversation_styles is None:
        context = getattr(services.agent_loop, "context", None)
        services.conversation_styles = getattr(context, "conversation_styles", None)
    if services.conversation_styles is None:
        services.conversation_styles = ConversationStyleService(services.workspace)

    attachments = DashboardAttachmentStore(services.workspace / "uploads" / "dashboard")
    broker = DashboardChatBroker(
        services.event_bus,
        services.agent_loop,
        attachments,
    )
    document_parser = DashboardDocumentParser(services.tools)
    if services.push_tool is not None:
        try:
            services.push_tool.register_channel(
                "dashboard",
                text=broker.push_text,
                file=broker.push_file,
            )
        except TypeError:
            # Compatibility for minimal test/third-party push registries that
            # still expose the historical text-only signature.
            services.push_tool.register_channel("dashboard", text=broker.push_text)
    app.state.dashboard_runtime = services
    app.state.dashboard_chat_broker = broker
    app.state.dashboard_attachment_store = attachments

    register_system_routes(app, services)
    register_model_routes(app, services)
    register_chat_routes(app, services, broker, attachments, document_parser)
    register_personal_routes(app, services)
    register_source_routes(app, services)
    register_rhythm_routes(app, services)
    register_memory_routes(app, services)
    register_attention_routes(app, services)
    register_workflow_routes(app, services)
