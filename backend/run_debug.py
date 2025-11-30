import os
import sys

import uvicorn
from dotenv import load_dotenv


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def main() -> None:
    ## Ensure '-Xfrozen_modules=off' by re-execing the interpreter once if needed
    ## Set a guard env var to avoid infinite recursion.
    #if os.environ.get("PY_FROZEN_OFF_APPLIED") != "1":
    #    os.environ["PY_FROZEN_OFF_APPLIED"] = "1"
    #    os.execv(
    #        sys.executable,
    #        [sys.executable, "-X", "frozen_modules=off", os.path.abspath(__file__)]
    #        + sys.argv[1:],
    #    )

    # Load environment variables from .env file
    load_dotenv()

    # Settings (overridable via env vars)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5001"))
    debug_host = os.environ.get("DEBUG_HOST", "127.0.0.1")
    debug_port = int(os.environ.get("DEBUG_PORT", "5678"))
    log_level = os.environ.get("LOG_LEVEL", "debug")
    wait_for_client = str_to_bool(os.environ.get("DEBUGPY_WAIT_FOR_CLIENT", "1"))

    # Which app to run (default to Binance testbed; override if needed)
    app_module = os.environ.get("APP_MODULE", "app.main_binance:app")

    # Start debugpy server (no reload)
    try:
        import debugpy  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "debugpy가 설치되어 있어야 합니다. `pip install debugpy` 후 다시 실행하세요."
        ) from exc

    debugpy.listen((debug_host, debug_port))
    if wait_for_client:
        # Attach 전까지 대기(옵션): DEBUGPY_WAIT_FOR_CLIENT=1 로 활성화
        debugpy.wait_for_client()

    # Hint for '-Xfrozen_modules=off'
    # 권장 실행: python -Xfrozen_modules=off backend/run_debug.py
    # (이 플래그는 런타임에서 변경할 수 없어, 실행 시 옵션으로 넘겨야 합니다)

    # Run uvicorn without reload/workers for stable debugging
    uvicorn.run(
        app_module,
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

