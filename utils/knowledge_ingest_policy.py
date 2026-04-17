from typing import Any, Dict


def should_treat_batch_duplicate(md5_hex: str, batch_md5s: set[str]) -> bool:
    """判断文件是否与当前上传批次中的文件重复。"""
    return bool(md5_hex) and (md5_hex in batch_md5s)


def can_retry_failed_meta(meta: Any) -> bool:
    """判断 ingest_status 条目是否为可安全读取的 failed 状态。"""
    if not isinstance(meta, dict):
        return False
    return str(meta.get("status", "")).strip() == "failed"

