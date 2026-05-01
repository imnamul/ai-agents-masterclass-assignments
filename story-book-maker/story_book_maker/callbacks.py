from datetime import datetime
from google.adk.agents.callback_context import CallbackContext


def before_agent_callback(callback_context: CallbackContext) -> None:
    print(f"[{datetime.now():%H:%M:%S}] START  {callback_context.agent_name}")


def after_agent_callback(callback_context: CallbackContext) -> None:
    print(f"[{datetime.now():%H:%M:%S}] DONE   {callback_context.agent_name}")
