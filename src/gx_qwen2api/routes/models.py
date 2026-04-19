"""GET /v1/models."""

from fastapi import APIRouter, Header, Request

from ..models import MODELS

router = APIRouter()


@router.get("/v1/models")
async def list_models(
    request: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> dict[str, str | list[dict[str, str | int]]]:
    from ..main import validate_api_key
    validate_api_key(x_api_key, authorization)
    data = list(MODELS)
    freebuff = getattr(request.app.state, "freebuff", None)
    if freebuff and freebuff.has_accounts():
        existing = {m["id"] for m in data if isinstance(m, dict) and "id" in m}
        for model in freebuff.list_models_payload():
            if model["id"] not in existing:
                data.append(model)
    return {"object": "list", "data": data}
