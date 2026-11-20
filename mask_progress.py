"""Report mask-specific frame progress to the requesting ComfyUI client."""

from time import monotonic


def frame_progress(phase, total):
    """Return a throttled reporter, with no ComfyUI dependency in CPU tests."""
    try:
        from comfy_execution.utils import get_executing_context
    except ImportError:
        return lambda _value: None

    context = get_executing_context()
    if context is None or not context.prompt_id or context.node_id is None:
        return lambda _value: None
    try:
        from server import PromptServer
    except ImportError:
        return lambda _value: None

    server = getattr(PromptServer, "instance", None)
    client_id = getattr(server, "client_id", None)
    if not client_id:
        return lambda _value: None

    payload = {"phase": phase, "max": int(total), "node": str(context.node_id), "prompt_id": str(context.prompt_id)}
    last_sent = None

    def report(value):
        nonlocal last_sent
        now = monotonic()
        if last_sent is not None and value < total and now - last_sent < 0.15:
            return
        last_sent = now
        try:
            server.send_sync("universal_inpaint_mask_progress", {**payload, "value": int(value)}, client_id)
        except RuntimeError:
            # The websocket event loop may already be closing.
            pass

    return report
