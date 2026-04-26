"""DeepSeek Proof-of-Work challenge solver using DeepSeekHashV1 WASM.

Downloads the DeepSeek sha3 WASM binary and calls wasm_solve through
wasmtime-py, mirroring the ds-free-api pow.rs implementation:

  prefix = f"{salt}_{expire_at}_"
  wasm_solve(retptr, challenge_ptr, challenge_len, prefix_ptr, prefix_len, difficulty)
  answer = read f64 from retptr+8 if status (i32 at retptr) != 0
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Any

logger = logging.getLogger("gx_qwen2api.deepseek.pow")

_WASM_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"

_solver_instance: PowSolver | None = None
_solver_lock: Any = None  # asyncio.Lock, set lazily


def _get_solver_lock() -> Any:
    global _solver_lock
    if _solver_lock is None:
        import asyncio
        _solver_lock = asyncio.Lock()
    return _solver_lock


class PowSolver:
    """WASM-based DeepSeekHashV1 PoW solver.

    Uses wasmtime-py to load and execute the DeepSeek WASM binary that
    implements the actual hash algorithm the server expects.
    """

    def __init__(self, wasm_bytes: bytes) -> None:
        import wasmtime

        self._engine = wasmtime.Engine()
        self._module = wasmtime.Module(self._engine, wasm_bytes)
        self._linker = wasmtime.Linker(self._engine)

        exports = list(self._module.exports)

        self._add_to_stack_name = self._find_export_by_names(
            exports, ["__wbindgen_add_to_stack_pointer"], [wasmtime.ValType.i32()], [wasmtime.ValType.i32()],
        )
        if not self._add_to_stack_name:
            raise PowError("__wbindgen_add_to_stack_pointer not found in WASM exports")

        self._alloc_name = (
            self._find_export_by_names(
                exports, ["__wbindgen_malloc"],
                [wasmtime.ValType.i32(), wasmtime.ValType.i32()],
                [wasmtime.ValType.i32()],
            )
            or self._find_export_by_prefix(
                exports, "__wbindgen_export_",
                [wasmtime.ValType.i32(), wasmtime.ValType.i32()],
                [wasmtime.ValType.i32()],
            )
        )
        if not self._alloc_name:
            raise PowError("allocator export not found in WASM exports")

        self._solve_name = (
            self._find_export_by_names(
                exports, ["wasm_solve"],
                [
                    wasmtime.ValType.i32(), wasmtime.ValType.i32(),
                    wasmtime.ValType.i32(), wasmtime.ValType.i32(),
                    wasmtime.ValType.i32(), wasmtime.ValType.f64(),
                ],
                [],
            )
            or self._find_solve_by_signature(exports)
        )
        if not self._solve_name:
            raise PowError("wasm_solve export not found in WASM exports")

    def solve(self, challenge_data: dict[str, Any]) -> int | None:
        """Solve a DeepSeekHashV1 challenge.

        Args:
            challenge_data: Dict with keys: algorithm, challenge, salt,
                signature, difficulty, expire_at, target_path.

        Returns:
            The answer as an int, or None if no solution found.
        """
        import wasmtime

        algorithm = challenge_data.get("algorithm", "")
        if algorithm != "DeepSeekHashV1":
            logger.error("Unsupported PoW algorithm: %s", algorithm)
            return None

        challenge_str = challenge_data.get("challenge", "")
        salt = challenge_data.get("salt", "")
        expire_at = challenge_data.get("expire_at", 0)
        difficulty = challenge_data.get("difficulty", 4)

        prefix = f"{salt}_{expire_at}_"

        store = wasmtime.Store(self._engine)
        instance = self._linker.instantiate(store, self._module)

        exports = instance.exports(store)

        memory = exports["memory"]
        if memory is None:
            logger.error("WASM memory export not found")
            return None

        add_to_stack = exports[self._add_to_stack_name]
        alloc_func = exports[self._alloc_name]
        solve_func = exports[self._solve_name]

        if add_to_stack is None or alloc_func is None or solve_func is None:
            logger.error("WASM required functions not found in instance")
            return None

        retptr_val = add_to_stack(store, -16)

        challenge_bytes = challenge_str.encode("utf-8")
        prefix_bytes = prefix.encode("utf-8")

        ptr_challenge_val = alloc_func(store, len(challenge_bytes), 1)
        ptr_prefix_val = alloc_func(store, len(prefix_bytes), 1)

        mem_buf = memory.data_ptr(store)

        def _write_mem(offset: int, data: bytes) -> None:
            for i, b in enumerate(data):
                mem_buf[offset + i] = b

        _write_mem(ptr_challenge_val, challenge_bytes)
        _write_mem(ptr_prefix_val, prefix_bytes)

        solve_func(
            store,
            retptr_val,
            ptr_challenge_val,
            len(challenge_bytes),
            ptr_prefix_val,
            len(prefix_bytes),
            float(difficulty),
        )

        status_buf = bytearray(4)
        for i in range(4):
            status_buf[i] = mem_buf[retptr_val + i]
        status = struct.unpack("<i", status_buf)[0]

        value_buf = bytearray(8)
        for i in range(8):
            value_buf[i] = mem_buf[retptr_val + 8 + i]
        value = struct.unpack("<d", value_buf)[0]

        add_to_stack(store, 16)

        if status == 0:
            logger.warning("WASM PoW solver returned status=0 (no solution)")
            return None

        answer = int(value)
        logger.debug("PoW solved: answer=%d (difficulty=%d)", answer, difficulty)
        return answer

    @staticmethod
    def _find_export_by_names(
        exports: list, names: list[str], param_types: list, result_types: list,
    ) -> str | None:
        for name in names:
            for exp in exports:
                if exp.name == name:
                    return name
        return None

    @staticmethod
    def _find_export_by_prefix(
        exports: list, prefix: str, param_types: list, result_types: list,
    ) -> str | None:
        for exp in exports:
            if exp.name.startswith(prefix):
                return exp.name
        return None

    @staticmethod
    def _find_solve_by_signature(exports: list) -> str | None:
        import wasmtime
        target_params = [
            wasmtime.ValType.i32(), wasmtime.ValType.i32(),
            wasmtime.ValType.i32(), wasmtime.ValType.i32(),
            wasmtime.ValType.i32(), wasmtime.ValType.f64(),
        ]
        candidates = []
        for exp in exports:
            if exp.type.kind == wasmtime.ExternKind.FUNC:
                try:
                    func_type = exp.type.func_type
                    if func_type is None:
                        continue
                    params = list(func_type.params)
                    results = list(func_type.results)
                    if len(params) == 6 and len(results) == 0:
                        candidates.append(exp.name)
                except Exception:
                    continue
        if len(candidates) == 1:
            return candidates[0]
        return None


class PowError(Exception):
    pass


async def get_solver(client: Any = None) -> PowSolver | None:
    """Get or create the singleton PowSolver, downloading WASM if needed."""
    global _solver_instance

    if _solver_instance is not None:
        return _solver_instance

    async with _get_solver_lock():
        if _solver_instance is not None:
            return _solver_instance

        try:
            import wasmtime
        except ImportError:
            logger.error("wasmtime package not installed; run: pip install wasmtime")
            return None

        wasm_bytes = await _download_wasm(client)
        if not wasm_bytes:
            return None

        try:
            _solver_instance = PowSolver(wasm_bytes)
            logger.info("DeepSeek PoW WASM solver initialized successfully")
            return _solver_instance
        except PowError as exc:
            logger.error("Failed to initialize PoW WASM solver: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error initializing PoW WASM solver: %s", exc)
            return None


async def _download_wasm(client: Any = None) -> bytes | None:
    """Download the DeepSeek WASM binary."""
    import httpx

    url = _WASM_URL
    logger.info("Downloading DeepSeek PoW WASM from %s", url)

    try:
        if client is not None and isinstance(client, httpx.AsyncClient):
            resp = await client.get(url, timeout=30.0)
            if resp.status_code == 200:
                return resp.content
            logger.warning("WASM download returned status %d", resp.status_code)

        async with httpx.AsyncClient() as fallback:
            resp = await fallback.get(url, timeout=30.0)
            if resp.status_code == 200:
                return resp.content
            logger.warning("WASM download returned status %d", resp.status_code)
    except Exception as exc:
        logger.error("Failed to download WASM: %s", exc)

    return None


def solve_pow_challenge(challenge_data: dict[str, Any]) -> int | None:
    """Synchronous solve using the initialized WASM solver.

    Must be called after get_solver() has been successfully called.
    Falls back to a pure-Python brute-force if WASM is not available.
    """
    if _solver_instance is not None:
        return _solver_instance.solve(challenge_data)

    logger.warning("WASM solver not available, falling back to brute-force (likely to be rejected)")
    return _brute_force_fallback(
        challenge_data.get("challenge", ""),
        challenge_data.get("salt", ""),
        challenge_data.get("difficulty", 4),
        challenge_data.get("expire_at", 0),
    )


def _brute_force_fallback(
    challenge: str, salt: str, difficulty: int, expire_at: int, max_iterations: int = 10_000_000,
) -> int | None:
    """Last-resort brute-force solver (likely incorrect for DeepSeekHashV1)."""
    import hashlib

    prefix = f"{salt}_{expire_at}_"
    target_prefix = "0" * difficulty
    prefix_bytes = prefix.encode("utf-8")

    for i in range(max_iterations):
        answer = str(i)
        digest = hashlib.sha256(prefix_bytes + answer.encode("utf-8")).hexdigest()
        if digest.startswith(target_prefix):
            logger.debug("Brute-force PoW solved after %d iterations", i)
            return i

    logger.warning("Brute-force PoW not solved within %d iterations", max_iterations)
    return None
