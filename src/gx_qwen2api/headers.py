"""DashScope request headers that mimic the Qwen Code client."""

USER_AGENT = "QwenCode/0.14.0 (linux; x64)"


def build_headers(access_token: str, *, streaming: bool = False) -> dict[str, str]:
    return {
        "connection": "keep-alive",
        "accept": "text/event-stream" if streaming else "application/json",
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "user-agent": USER_AGENT,
        "x-dashscope-authtype": "qwen-oauth",
        "x-dashscope-cachecontrol": "enable",
        "x-dashscope-useragent": USER_AGENT,
        "x-stainless-arch": "x64",
        "x-stainless-lang": "js",
        "x-stainless-os": "Linux",
        "x-stainless-package-version": "5.11.0",
        "x-stainless-retry-count": "0",
        "x-stainless-runtime": "node",
        "x-stainless-runtime-version": "v25.7.0",
        "accept-language": "*",
        "sec-fetch-mode": "cors",
        "accept-encoding": "br, gzip, deflate",
    }
