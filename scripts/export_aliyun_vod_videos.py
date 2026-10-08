#!/usr/bin/env python3
"""从阿里云 VOD 批量导出媒资，生成 Push 可用的 JSON + 审查用 CSV。

依赖（一次性安装）:
    pip install -r push-system/requirements-vod-export.txt

环境变量（写入仓库根 .env 或 export）:
    ALIYUN_ACCESS_KEY_ID
    ALIYUN_ACCESS_KEY_SECRET
    ALIYUN_VOD_REGION=cn-shanghai   # 与 VOD 控制台地域一致

用法:
    python scripts/export_aliyun_vod_videos.py
    python scripts/export_aliyun_vod_videos.py --output push-system/data/aliyun_videos.json
    python scripts/export_aliyun_vod_videos.py --csv-only
    python scripts/export_aliyun_vod_videos.py --dry-run

导出后重建 Push 向量（元数据变更时）:
    cd push-system && python -m data.generate_data
    或 scripts\\setup_push_data.bat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_data_dir() -> Path:
    """Repo checkout (push-system/data) or cloud deploy bundle (./data)."""
    deploy_data = ROOT / "data"
    if deploy_data.is_dir() and (ROOT / "server").is_dir():
        return deploy_data
    return ROOT / "push-system" / "data"


DATA_DIR = _resolve_data_dir()
DEFAULT_OUTPUT = DATA_DIR / "aliyun_videos.json"
DEFAULT_CSV = DATA_DIR / "aliyun_videos.csv"
DEFAULT_RAW = DATA_DIR / "aliyun_videos_raw.json"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _require_sdk():
    try:
        from aliyunsdkcore.client import AcsClient  # noqa: F401
        from aliyunsdkvod.request.v20170321 import (  # noqa: F401
            GetVideoInfosRequest,
            SearchMediaRequest,
        )
    except ImportError as exc:
        print(
            "[ERROR] 缺少阿里云 SDK。请执行:\n"
            "  pip install -r push-system/requirements-vod-export.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _get_credentials() -> tuple[str, str, str]:
    _load_dotenv()
    ak = os.environ.get("ALIYUN_ACCESS_KEY_ID", "").strip()
    sk = os.environ.get("ALIYUN_ACCESS_KEY_SECRET", "").strip()
    region = os.environ.get("ALIYUN_VOD_REGION", "cn-shanghai").strip() or "cn-shanghai"
    if not ak or not sk:
        print(
            "[ERROR] 请在 .env 中配置 ALIYUN_ACCESS_KEY_ID 与 ALIYUN_ACCESS_KEY_SECRET",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return ak, sk, region


def _title_sort_key(title: str, creation_time: str) -> tuple:
    m = re.match(r"^\s*(\d+)\.", title or "")
    if m:
        return (0, int(m.group(1)), title)
    return (1, creation_time or "", title)


def _parse_tags(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[,，;；\s]+", str(raw).strip())
    return [p for p in (x.strip() for x in parts) if p]


def _make_summary(description: str, title: str, max_len: int = 48) -> str:
    text = (description or "").strip() or (title or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def search_all_videos(client, *, page_size: int = 100) -> list[dict]:
    from aliyunsdkvod.request.v20170321 import SearchMediaRequest

    all_items: list[dict] = []
    page_no = 1
    while True:
        req = SearchMediaRequest.SearchMediaRequest()
        req.set_accept_format("JSON")
        req.set_SearchType("video")
        req.set_Match('Status in ("Normal")')
        req.set_Fields(
            "VideoId,Title,CoverURL,Duration,Status,CreationTime,Size,StorageLocation"
        )
        req.set_SortBy("CreationTime:Asc")
        req.set_PageNo(page_no)
        req.set_PageSize(page_size)

        raw = client.do_action_with_exception(req)
        payload = json.loads(raw)
        media_list = payload.get("MediaList") or []
        if not media_list:
            break
        all_items.extend(media_list)
        total = int(payload.get("Total") or 0)
        print(f"  SearchMedia page {page_no}: +{len(media_list)} (total fetched {len(all_items)}/{total or '?'})")
        if total and len(all_items) >= total:
            break
        if len(media_list) < page_size:
            break
        page_no += 1
    return all_items


def fetch_video_details(client, video_ids: list[str]) -> dict[str, dict]:
    from aliyunsdkvod.request.v20170321 import GetVideoInfosRequest

    details: dict[str, dict] = {}
    chunk_size = 20
    for i in range(0, len(video_ids), chunk_size):
        chunk = video_ids[i : i + chunk_size]
        req = GetVideoInfosRequest.GetVideoInfosRequest()
        req.set_accept_format("JSON")
        req.set_VideoIds(",".join(chunk))
        raw = client.do_action_with_exception(req)
        payload = json.loads(raw)
        for item in payload.get("VideoList") or []:
            vid = item.get("VideoId")
            if vid:
                details[vid] = item
    return details


def _parse_duration(value) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if text.isdigit():
        return float(text)
    if ":" in text:
        parts = text.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return 0.0
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _pick_field(row: dict, *names: str) -> str:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        key = name.lower()
        if key in lowered and lowered[key] not in (None, ""):
            return str(lowered[key]).strip()
    return ""


def import_from_vod_csv(csv_path: Path) -> list[dict]:
    """从 VOD 控制台「导出」的 CSV/TSV 转为 Push 元数据（无需 API Key）。"""
    text = csv_path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if text.count("\t") > text.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows:
        raise ValueError(f"CSV 为空: {csv_path}")

    merged: list[dict] = []
    for row in rows:
        vid = _pick_field(row, "VideoId", "videoId", "video_id", "媒资ID", "媒资 id", "ID", "id")
        title = _pick_field(row, "Title", "title", "名称", "视频名称", "name")
        if not vid and not title:
            continue
        if not vid:
            vid = title
        cover = _pick_field(row, "CoverURL", "cover_url", "封面URL", "封面", "Cover")
        desc = _pick_field(row, "Description", "description", "描述", "简介")
        tags_raw = _pick_field(row, "Tags", "tags", "标签")
        duration_raw = _pick_field(row, "Duration", "duration", "时长", "DurationSec") or "0"
        creation = _pick_field(row, "CreationTime", "creation_time", "创建时间", "CreateTime")
        merged.append(
            {
                "vod_video_id": vid,
                "title": title,
                "description": desc,
                "tags": _parse_tags(tags_raw),
                "cover_url": cover,
                "duration_sec": round(_parse_duration(duration_raw), 3),
                "creation_time": creation,
            }
        )

    merged.sort(key=lambda x: _title_sort_key(x["title"], x.get("creation_time", "")))
    out: list[dict] = []
    for idx, item in enumerate(merged, start=1):
        out.append(
            {
                "video_id": f"v_{idx:03d}",
                "vod_video_id": item["vod_video_id"],
                "title": item["title"],
                "description": item["description"],
                "summary": _make_summary(item["description"], item["title"]),
                "tags": item["tags"],
                "cover_url": item["cover_url"],
                "duration_sec": item["duration_sec"],
            }
        )
    return out


def build_push_records(search_items: list[dict], details: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for row in search_items:
        vid = row.get("VideoId") or row.get("MediaId")
        if not vid:
            continue
        info = details.get(vid, {})
        title = (info.get("Title") or row.get("Title") or "").strip()
        description = (info.get("Description") or "").strip()
        cover = (info.get("CoverURL") or row.get("CoverURL") or "").strip()
        duration = float(info.get("Duration") or row.get("Duration") or 0)
        creation = info.get("CreationTime") or row.get("CreationTime") or ""
        tags = _parse_tags(info.get("Tags"))
        merged.append(
            {
                "vod_video_id": vid,
                "title": title,
                "description": description,
                "tags": tags,
                "cover_url": cover,
                "duration_sec": round(duration, 3),
                "creation_time": creation,
                "_size_bytes": info.get("Size") or row.get("Size"),
                "_storage": info.get("StorageLocation") or row.get("StorageLocation"),
            }
        )

    merged.sort(key=lambda x: _title_sort_key(x["title"], x.get("creation_time", "")))

    out: list[dict] = []
    for idx, item in enumerate(merged, start=1):
        out.append(
            {
                "video_id": f"v_{idx:03d}",
                "vod_video_id": item["vod_video_id"],
                "title": item["title"],
                "description": item["description"],
                "summary": _make_summary(item["description"], item["title"]),
                "tags": item["tags"],
                "cover_url": item["cover_url"],
                "duration_sec": item["duration_sec"],
            }
        )
    return out


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "vod_video_id",
        "title",
        "duration_sec",
        "cover_url",
        "description",
        "tags",
        "summary",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            r = dict(row)
            r["tags"] = "|".join(row.get("tags") or [])
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导出阿里云 VOD 媒资为 Push 元数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Push 格式 JSON 输出路径（默认 {DEFAULT_OUTPUT.relative_to(ROOT)}）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV 审查文件路径",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW,
        help="原始 SearchMedia + GetVideoInfos 合并 JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="只拉取统计，不写文件")
    parser.add_argument("--csv-only", action="store_true", help="仅写 CSV（仍拉 API）")
    parser.add_argument("--no-details", action="store_true", help="跳过 GetVideoInfos（无描述/标签）")
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="从 VOD 控制台导出的 CSV/TSV 转换（无需 API Key）",
    )
    args = parser.parse_args()

    if args.from_csv:
        print(f"[import] 从 CSV 转换: {args.from_csv}")
        records = import_from_vod_csv(args.from_csv.resolve())
        missing_cover = sum(1 for r in records if not r.get("cover_url"))
        missing_desc = sum(1 for r in records if not r.get("description"))
        print(f"  共 {len(records)} 条 · 缺封面 {missing_cover} · 缺描述 {missing_desc}")
        if args.dry_run:
            for r in records[:5]:
                print(f"  {r['video_id']} | {r['vod_video_id'][:12]}… | {r['title'][:40]}")
            return 0
        if not args.csv_only:
            write_json(args.output, records)
            print(f"  JSON -> {args.output}")
        write_csv(args.csv, records)
        print(f"  CSV  -> {args.csv}")
        return 0

    _require_sdk()
    from aliyunsdkcore.client import AcsClient

    ak, sk, region = _get_credentials()
    client = AcsClient(ak, sk, region)

    print(f"[1/3] SearchMedia region={region} ...")
    search_items = search_all_videos(client)
    if not search_items:
        print("[WARN] 未找到状态为 Normal 的视频")
        return 0

    video_ids = [x.get("VideoId") or x.get("MediaId") for x in search_items]
    video_ids = [v for v in video_ids if v]

    details: dict[str, dict] = {}
    if not args.no_details:
        print(f"[2/3] GetVideoInfos x{len(video_ids)} ...")
        details = fetch_video_details(client, video_ids)
    else:
        print("[2/3] Skip GetVideoInfos (--no-details)")

    records = build_push_records(search_items, details)
    missing_cover = sum(1 for r in records if not r.get("cover_url"))
    missing_desc = sum(1 for r in records if not r.get("description"))

    print(f"[3/3] 共 {len(records)} 条 · 缺封面 {missing_cover} · 缺描述 {missing_desc}")

    if args.dry_run:
        for r in records[:5]:
            print(f"  {r['video_id']} | {r['vod_video_id'][:12]}… | {r['title'][:40]}")
        if len(records) > 5:
            print(f"  ... 还有 {len(records) - 5} 条")
        return 0

    if not args.csv_only:
        write_json(args.output, records)
        print(f"  JSON -> {args.output}")

    write_csv(args.csv, records)
    print(f"  CSV  -> {args.csv}")

    raw_bundle = {
        "region": region,
        "count": len(records),
        "search_media": search_items,
        "video_infos": details,
        "push_records": records,
    }
    write_json(args.raw, raw_bundle)
    print(f"  RAW  -> {args.raw}")

    print("\n下一步（替换 Push 元数据并重建向量）:")
    print(f"  1. 检查 CSV，在 VOD 控制台补全缺失的描述/标签后重新导出")
    print(f"  2. 确认 JSON 无误后: copy / 设置 PUSH_VIDEOS={args.output.name}")
    print("  3. cd push-system && python -m data.generate_data")
    print("  4. 重新打 Push 云包: scripts\\package_push_cloud_split.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
