from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_MAX_QUOTA_WAITS = 100
DEFAULT_QUOTA_BUFFER = 5.0
DEFAULT_QUOTA_RETRY_DELAY = 30.0
DEFAULT_IMAGE_MAX_SIDE = 1800
DEFAULT_OCR_TIMEOUT = 180.0
DEFAULT_SKIP_AFTER = 20.0
DEFAULT_PARTY_LIST_SKIP_AFTER = 90.0
ENV_PATH = Path(__file__).resolve().parent / ".env"


@dataclass
class TemplateRow:
    id: str
    prefix: str
    ballot_number: int


def load_dotenv_file(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_gray_image(image_path: Path):
    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def threshold_image(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def build_line_masks(image) -> list[tuple[object, object]]:
    height, width = image.shape[:2]
    threshold = threshold_image(image)
    kernel_pairs = [
        (max(40, min(160, width // 18)), max(40, min(160, height // 18))),
        (max(32, min(140, width // 24)), max(32, min(140, height // 24))),
        (max(24, min(120, width // 32)), max(24, min(120, height // 32))),
    ]

    masks: list[tuple[object, object]] = []
    seen: set[tuple[int, int]] = set()
    for horizontal_size, vertical_size in kernel_pairs:
        key = (horizontal_size, vertical_size)
        if key in seen:
            continue
        seen.add(key)
        horizontal = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1)),
        )
        vertical = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size)),
        )
        masks.append((horizontal, vertical))
    return masks


def score_table_candidate(
    bbox: tuple[int, int, int, int],
    horizontal,
    vertical,
    image_shape: tuple[int, int],
) -> float:
    x, y, w, h = bbox
    image_height, image_width = image_shape[:2]
    if w <= 0 or h <= 0:
        return -1.0
    if w / image_width < 0.2 or h / image_height < 0.03:
        return -1.0

    horizontal_crop = horizontal[y : y + h, x : x + w]
    vertical_crop = vertical[y : y + h, x : x + w]
    horizontal_pixels = cv2.countNonZero(horizontal_crop)
    vertical_pixels = cv2.countNonZero(vertical_crop)
    if horizontal_pixels < 3000 or vertical_pixels < 1500:
        return -1.0

    area = w * h
    density = (horizontal_pixels + vertical_pixels) / max(1, area)
    bottom_ratio = (y + h) / image_height
    return area + min(horizontal_pixels, vertical_pixels) * 8 + density * 500000 + bottom_ratio * 25000


def find_table_bbox(image_path: Path):
    image = load_gray_image(image_path)
    if image is None:
        return None

    best_bbox = None
    best_score = -1.0
    for horizontal, vertical in build_line_masks(image):
        mask = horizontal | vertical
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if int(area) < 8000:
                continue
            bbox = (int(x), int(y), int(w), int(h))
            score = score_table_candidate(bbox, horizontal, vertical, image.shape)
            if score > best_score:
                best_bbox = bbox
                best_score = score
    return best_bbox


def cache_dir_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".cache")


def prepare_typhoon_image(
    image_path: Path,
    cache_dir: Path,
    *,
    max_side: int = DEFAULT_IMAGE_MAX_SIDE,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = cache_dir / f"{image_path.stem}.prepared.png"
    if prepared_path.exists():
        return prepared_path

    image = load_gray_image(image_path)
    if image is None:
        raise RuntimeError(f"cannot read image: {image_path}")

    bbox = find_table_bbox(image_path)
    if bbox is not None:
        x, y, w, h = bbox
        pad_x = max(20, int(w * 0.03))
        pad_y_top = max(20, int(h * 0.08))
        pad_y_bottom = max(20, int(h * 0.03))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y_bottom)
        image = image[y1:y2, x1:x2]

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_side:
        scale = max_side / longest_side
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    cv2.imwrite(str(prepared_path), image)
    return prepared_path


def load_template(template_path: Path) -> dict[str, list[TemplateRow]]:
    groups: dict[str, list[TemplateRow]] = defaultdict(list)
    with template_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["id"].strip()
            prefix, ballot_text = identifier.rsplit("_", 1)
            groups[prefix].append(
                TemplateRow(id=identifier, prefix=prefix, ballot_number=int(ballot_text))
            )
    for prefix in groups:
        groups[prefix].sort(key=lambda item: item.ballot_number)
    return dict(groups)

def resolve_api_key(cli_api_key: str | None = None) -> str | None:
    load_dotenv_file()
    if cli_api_key:
        return cli_api_key
    for env_name in [
        "TYPHOON_OCR_API_KEY",
        "TYPHOON_API_KEY",
        "OPENAI_API_KEY",
    ]:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def page_group_key(path: Path) -> str:
    return re.sub(r"_page\d+$", "", path.stem)


def page_number(path: Path) -> int:
    match = re.search(r"_page(\d+)$", path.stem)
    return int(match.group(1)) if match else 1


def is_party_list_page(path: Path) -> bool:
    return page_group_key(path).startswith("party_list_")


def effective_skip_after_for_page(
    image_path: Path,
    skip_after: float | None,
    party_list_skip_after: float | None,
) -> float | None:
    if skip_after is None:
        return None
    if is_party_list_page(image_path):
        if party_list_skip_after is None:
            return None
        return max(skip_after, party_list_skip_after)
    return skip_after


def numeric_sort_key(path: Path) -> tuple:
    parts = re.findall(r"\d+|\D+", path.stem)
    normalized = []
    for part in parts:
        normalized.append(int(part) if part.isdigit() else part)
    return tuple(normalized)


def normalize_digits(text: str) -> str:
    return text.translate(THAI_DIGITS)


def progress_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".progress.json")


def skipped_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".skipped.txt")


def missing_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".missing.csv")


def load_existing_votes(output_path: Path) -> dict[str, int]:
    if not output_path.exists():
        return {}

    votes_by_id: dict[str, int] = {}
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["id"].strip()
            try:
                votes_by_id[identifier] = int(str(row["votes"]).strip() or "0")
            except ValueError:
                votes_by_id[identifier] = 0
    return votes_by_id


def append_skipped_page(skipped_path: Path, page_key: str, reason: str) -> None:
    existing: list[str] = []
    seen = False
    if skipped_path.exists():
        existing = skipped_path.read_text(encoding="utf-8").splitlines()
        for line in existing:
            if line.split("\t", 1)[0] == page_key:
                seen = True
                break
    if seen:
        return
    with skipped_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{page_key}\t{reason}\n")


def load_skipped_pages(skipped_path: Path) -> set[str]:
    if not skipped_path.exists():
        return set()
    skipped_pages: set[str] = set()
    for line in skipped_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        skipped_pages.add(line.split("\t", 1)[0])
    return skipped_pages


def rewrite_skipped_pages(skipped_path: Path, skipped_pages: set[str]) -> None:
    if not skipped_pages:
        if skipped_path.exists():
            skipped_path.unlink()
        return
    skipped_path.write_text(
        "".join(f"{page_key}\tparsed_0_rows_or_timeout\n" for page_key in sorted(skipped_pages)),
        encoding="utf-8",
    )


def load_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    completed_pages = payload.get("completed_pages", [])
    return {str(item) for item in completed_pages}


def write_progress(progress_path: Path, completed_pages: set[str]) -> None:
    payload = {
        "completed_pages": sorted(completed_pages),
        "updated_at": int(time.time()),
    }
    progress_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def apply_start_filters(
    pages: list[Path],
    *,
    start_from: str | None = None,
    start_index: int | None = None,
) -> list[Path]:
    ordered_pages = sorted(pages, key=lambda path: (numeric_sort_key(path), page_number(path), path.name))
    if start_from:
        start_path = Path(start_from)
        start_name = start_path.name
        start_stem = start_path.stem
        for index, page in enumerate(ordered_pages):
            if (
                page.name == start_from
                or page.stem == start_from
                or page.name == start_name
                or page.stem == start_stem
                or str(page) == start_from
            ):
                return ordered_pages[index:]
        raise RuntimeError(f"start image not found: {start_from}")
    if start_index is not None:
        if start_index < 1 or start_index > len(ordered_pages):
            raise RuntimeError(f"start index out of range: {start_index} (have {len(ordered_pages)} images)")
        return ordered_pages[start_index - 1 :]
    return ordered_pages


def resolve_start_page(
    pages: list[Path],
    *,
    start_from: str | None = None,
    start_index: int | None = None,
) -> Path | None:
    filtered = apply_start_filters(
        pages,
        start_from=start_from,
        start_index=start_index,
    )
    if not filtered:
        return None
    return filtered[0]


def extract_retry_delay_seconds(error: Exception, default_delay: float) -> float:
    message = str(error)
    retry_in_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if retry_in_match:
        return max(default_delay, float(retry_in_match.group(1)) + 1.0)
    retry_delay_match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s['\"]?", message)
    if retry_delay_match:
        return max(default_delay, float(retry_delay_match.group(1)) + 1.0)
    return default_delay


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "resource_exhausted" in message
        or "quota exceeded" in message
        or "current quota" in message
        or "rate limit" in message
        or "ratelimiterror" in message
        or "rate exceeded" in message
        or "429" in message
    )


def log_stage(page_name: str, stage: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{page_name}] {stage}{suffix}")


def typhoon_ocr_markdown(image_path: Path) -> str:
    try:
        from typhoon_ocr import ocr_document
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: typhoon-ocr. Install with `pip install typhoon-ocr` "
            "and set TYPHOON_OCR_API_KEY or OPENAI_API_KEY."
        ) from exc
    return ocr_document(str(image_path))


def _ocr_worker(image_path: str, queue) -> None:
    try:
        markdown = typhoon_ocr_markdown(Path(image_path))
        queue.put({"ok": True, "markdown": markdown})
    except Exception as exc:  # pragma: no cover
        queue.put({"ok": False, "error": repr(exc)})


def typhoon_ocr_markdown_with_timeout(image_path: Path, timeout_seconds: float) -> str:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_ocr_worker, args=(str(image_path), queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"OCR timeout after {timeout_seconds:.1f}s for {image_path.name}")

    if queue.empty():
        raise RuntimeError(f"OCR worker exited without result for {image_path.name}")

    result = queue.get()
    if result.get("ok"):
        return str(result["markdown"])
    raise RuntimeError(f"Typhoon OCR failed for {image_path.name}: {result.get('error', 'unknown error')}")


def typhoon_ocr_markdown_cached(
    image_path: Path,
    cache_dir: Path,
    *,
    timeout_seconds: float,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = cache_dir / f"{image_path.stem}.md"
    if markdown_path.exists():
        log_stage(image_path.name, "ocr-cache-hit", str(markdown_path.name))
        return markdown_path.read_text(encoding="utf-8")

    log_stage(image_path.name, "ocr-start")
    started = time.time()
    markdown = typhoon_ocr_markdown_with_timeout(image_path, timeout_seconds=timeout_seconds)
    markdown_path.write_text(markdown, encoding="utf-8")
    log_stage(image_path.name, "ocr-done", f"{time.time() - started:.1f}s")
    return markdown


def parse_vote_candidates(text: str) -> list[int]:
    normalized = normalize_digits(text)
    candidates: list[int] = []
    for token in re.findall(r"\d{1,3},\d{3}|\d{2,5}", normalized):
        value = int(token.replace(",", ""))
        if value > 0:
            candidates.append(value)
    return candidates


def clean_html_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_rows_from_html_table(markdown: str, expected_max_ballot: int) -> dict[int, int]:
    votes_by_ballot: dict[int, int] = {}
    row_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", markdown, flags=re.IGNORECASE | re.DOTALL)
    for row_html in row_matches:
        cells = [
            clean_html_text(cell_html)
            for cell_html in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        ]
        if len(cells) < 2:
            continue
        if any("รวมคะแนน" in cell or "หมายเลข" in cell or "ได้คะแนน" in cell for cell in cells):
            continue

        ballot_candidates = [
            int(token)
            for token in re.findall(r"\d+", normalize_digits(cells[0]))
            if 1 <= int(token) <= expected_max_ballot
        ]
        if not ballot_candidates:
            continue
        ballot_number = ballot_candidates[0]

        vote_cells = []
        for cell in cells[1:]:
            candidates = parse_vote_candidates(cell)
            if candidates:
                vote_cells.append((cell, candidates))
        if not vote_cells:
            continue

        _vote_cell, vote_candidates = max(
            vote_cells,
            key=lambda item: (
                max(len(str(value)) for value in item[1]),
                max(item[1]),
            ),
        )
        formatted = [value for value in vote_candidates if len(str(value)) >= 3]
        chosen = max(formatted) if formatted else max(vote_candidates)
        votes_by_ballot[ballot_number] = chosen

    return votes_by_ballot


def parse_rows_from_markdown(markdown: str, expected_max_ballot: int) -> dict[int, int]:
    html_rows = parse_rows_from_html_table(markdown, expected_max_ballot)
    if html_rows:
        return html_rows

    votes_by_ballot: dict[int, int] = {}
    for raw_line in markdown.splitlines():
        line = normalize_digits(raw_line).strip()
        if not line:
            continue
        if "รวมคะแนน" in raw_line or "หมายเลข" in raw_line or "ได้คะแนน" in raw_line:
            continue

        tokens = re.findall(r"\d{1,3},\d{3}|\d+", line)
        if len(tokens) < 2:
            continue

        ballot_number = None
        for token in tokens:
            value = int(token.replace(",", ""))
            if 1 <= value <= expected_max_ballot:
                ballot_number = value
                break
        if ballot_number is None:
            continue

        vote_candidates = [
            value
            for value in parse_vote_candidates(line)
            if value > ballot_number
        ]
        if not vote_candidates:
            continue

        formatted = [value for value in vote_candidates if len(str(value)) >= 3]
        chosen = max(formatted) if formatted else max(vote_candidates)
        votes_by_ballot[ballot_number] = chosen
    return votes_by_ballot


def extract_rows_from_page(
    image_path: Path,
    expected_max_ballot: int,
    *,
    cache_dir: Path,
    ocr_timeout: float,
) -> dict[int, int]:
    log_stage(image_path.name, "prepare-start")
    prepared_image_path = prepare_typhoon_image(image_path, cache_dir / "prepared")
    log_stage(image_path.name, "prepare-done", prepared_image_path.name)
    markdown = typhoon_ocr_markdown_cached(
        prepared_image_path,
        cache_dir / "markdown",
        timeout_seconds=ocr_timeout,
    )
    log_stage(image_path.name, "parse-start")
    rows = parse_rows_from_markdown(markdown, expected_max_ballot=expected_max_ballot)
    log_stage(image_path.name, "parse-done", f"{len(rows)} rows")
    return rows


def extract_rows_with_retry(
    image_path: Path,
    expected_max_ballot: int,
    *,
    cache_dir: Path,
    ocr_timeout: float,
    skip_after: float | None,
    max_retries: int,
    retry_delay: float,
    max_quota_waits: int,
    quota_buffer: float,
) -> dict[int, int]:
    last_error: Exception | None = None
    attempt = 1
    quota_wait_count = 0
    effective_timeout = (
        min(ocr_timeout, skip_after)
        if skip_after is not None
        else ocr_timeout
    )
    while attempt <= max_retries:
        try:
            return extract_rows_from_page(
                image_path=image_path,
                expected_max_ballot=expected_max_ballot,
                cache_dir=cache_dir,
                ocr_timeout=effective_timeout,
            )
        except Exception as exc:
            last_error = exc
            if isinstance(exc, TimeoutError) and skip_after is not None:
                print(f"skip {image_path.name}: exceeded {skip_after:.1f}s")
                return {}
            if is_quota_error(exc):
                quota_wait_count += 1
                if quota_wait_count > max_quota_waits:
                    break
                base_delay = max(
                    DEFAULT_QUOTA_RETRY_DELAY,
                    retry_delay * (2 ** max(0, quota_wait_count - 1)),
                )
                wait_seconds = extract_retry_delay_seconds(exc, base_delay) + quota_buffer
                print(
                    f"quota wait {quota_wait_count}/{max_quota_waits} on {image_path.name}: {exc} "
                    f"(sleep {wait_seconds:.1f}s)"
                )
                time.sleep(wait_seconds)
                continue

            if attempt >= max_retries:
                break
            wait_seconds = extract_retry_delay_seconds(exc, retry_delay * attempt)
            print(
                f"retry {attempt}/{max_retries - 1} after error on {image_path.name}: {exc} "
                f"(sleep {wait_seconds:.1f}s)"
            )
            time.sleep(wait_seconds)
            attempt += 1

    assert last_error is not None
    raise last_error


def write_submission(
    output_path: Path,
    template_groups: dict[str, list[TemplateRow]],
    votes_by_id: dict[str, int],
    prefixes: list[str],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "votes"])
        for prefix in prefixes:
            for row in template_groups[prefix]:
                writer.writerow([row.id, votes_by_id.get(row.id, 0)])


def write_missing_report(
    output_path: Path,
    template_groups: dict[str, list[TemplateRow]],
    votes_by_id: dict[str, int],
    prefixes: list[str],
) -> None:
    missing_path = missing_path_for(output_path)
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix", "id"])
        for prefix in prefixes:
            for row in template_groups[prefix]:
                if votes_by_id.get(row.id, 0) <= 0:
                    writer.writerow([prefix, row.id])


def find_incomplete_prefixes(
    template_groups: dict[str, list[TemplateRow]],
    votes_by_id: dict[str, int],
    prefixes: list[str],
) -> list[str]:
    incomplete: list[str] = []
    for prefix in prefixes:
        rows = template_groups.get(prefix, [])
        if any(votes_by_id.get(row.id, 0) <= 0 for row in rows):
            incomplete.append(prefix)
    return incomplete


def run_incremental(
    template_path: Path,
    images_dir: Path,
    output_path: Path,
    all_images_dir: Path | None = Path("images"),
    api_key: str | None = None,
    prefix_filter: set[str] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    max_quota_waits: int = DEFAULT_MAX_QUOTA_WAITS,
    quota_buffer: float = DEFAULT_QUOTA_BUFFER,
    ocr_timeout: float = DEFAULT_OCR_TIMEOUT,
    skip_after: float | None = DEFAULT_SKIP_AFTER,
    party_list_skip_after: float | None = DEFAULT_PARTY_LIST_SKIP_AFTER,
    start_from: str | None = None,
    start_index: int | None = None,
    reset_progress: bool = False,
    retry_skipped_only: bool = False,
    retry_incomplete_prefixes: bool = False,
) -> None:
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        raise RuntimeError(
            "Please set TYPHOON_OCR_API_KEY, TYPHOON_API_KEY, or OPENAI_API_KEY, "
            "or pass --api-key."
        )
    os.environ["OPENAI_API_KEY"] = resolved_api_key

    template_groups = load_template(template_path)
    image_groups: dict[str, list[Path]] = defaultdict(list)
    fallback_image_groups: dict[str, list[Path]] = defaultdict(list)
    all_pages: list[Path] = []
    for image_path in sorted(images_dir.glob("*.png")):
        prefix = page_group_key(image_path)
        if prefix_filter and prefix not in prefix_filter:
            continue
        image_groups[prefix].append(image_path)
        all_pages.append(image_path)
    if all_images_dir and all_images_dir.exists():
        for image_path in sorted(all_images_dir.glob("*.png")):
            prefix = page_group_key(image_path)
            if prefix_filter and prefix not in prefix_filter:
                continue
            fallback_image_groups[prefix].append(image_path)

    all_prefixes = sorted(prefix_filter) if prefix_filter else sorted(template_groups)
    prefixes = list(all_prefixes)
    progress_path = progress_path_for(output_path)
    skipped_path = skipped_path_for(output_path)
    cache_dir = cache_dir_for(output_path)
    skipped_pages = load_skipped_pages(skipped_path)
    if reset_progress:
        completed_pages = set()
        votes_by_id = {}
        skipped_pages = set()
        if progress_path.exists():
            progress_path.unlink()
        if skipped_path.exists():
            skipped_path.unlink()
        write_submission(output_path, template_groups, votes_by_id, all_prefixes)
        write_missing_report(output_path, template_groups, votes_by_id, all_prefixes)
        print(f"reset progress and restarted from first page in {output_path}")
    else:
        votes_by_id = load_existing_votes(output_path)
        completed_pages = load_progress(progress_path)
        existing_nonzero = sum(1 for value in votes_by_id.values() if value > 0)

        if existing_nonzero == 0 and completed_pages:
            print("existing output has only zeros; ignoring previous completed-page progress")
            completed_pages = set()
            write_progress(progress_path, completed_pages)

        if output_path.exists():
            print(f"resuming from existing {output_path}")
        else:
            write_submission(output_path, template_groups, votes_by_id, all_prefixes)
            write_missing_report(output_path, template_groups, votes_by_id, all_prefixes)
            print(f"initialized {output_path}")

    if retry_incomplete_prefixes:
        incomplete_prefixes = find_incomplete_prefixes(template_groups, votes_by_id, all_prefixes)
        print(
            f"retrying incomplete prefixes only: {len(incomplete_prefixes)}/{len(all_prefixes)} prefixes"
        )
        prefixes = incomplete_prefixes
    else:
        incomplete_prefixes = []

    start_page = resolve_start_page(
        all_pages,
        start_from=start_from,
        start_index=start_index,
    ) if (start_from or start_index is not None) else None
    start_prefix = page_group_key(start_page) if start_page else None
    start_page_name = start_page.name if start_page else None
    start_reached = start_page is None

    if start_page is not None:
        print(f"starting from {start_page.name}")

    for prefix in prefixes:
        rows = template_groups[prefix]
        pages = image_groups.get(prefix, [])
        if prefix in incomplete_prefixes and fallback_image_groups:
            selected_names = {page.name for page in pages}
            rescued_pages: list[Path] = []
            for candidate in fallback_image_groups.get(prefix, []):
                if candidate.name in selected_names:
                    continue
                if find_table_bbox(candidate) is None:
                    continue
                rescued_pages.append(candidate)
            if rescued_pages:
                pages = pages + rescued_pages
                pages = sorted(
                    pages,
                    key=lambda path: (numeric_sort_key(path), page_number(path), path.name),
                )
                print(
                    f"rescued {len(rescued_pages)} extra table pages from {all_images_dir} for {prefix}"
                )
        if not pages:
            print(f"skip {prefix}: no images")
            continue

        if not start_reached:
            if prefix != start_prefix:
                print(f"skip before start {prefix}")
                continue
            pages = apply_start_filters(pages, start_from=start_page_name)
            start_reached = True
        else:
            pages = sorted(pages, key=lambda path: (numeric_sort_key(path), page_number(path), path.name))

        total_written = 0
        force_rerun_prefix = prefix in incomplete_prefixes
        for page in pages:
            page_key = f"{prefix}/{page.name}"
            if retry_skipped_only and page_key not in skipped_pages:
                continue
            if page_key in completed_pages and not force_rerun_prefix:
                total_written = sum(1 for row in rows if votes_by_id.get(row.id, 0) > 0)
                print(f"skip completed {page.name} for {prefix} ({total_written}/{len(rows)})")
                continue
            if page_key in completed_pages and force_rerun_prefix:
                print(f"rerun incomplete {page.name} for {prefix}")

            page_skip_after = effective_skip_after_for_page(
                page,
                skip_after=skip_after,
                party_list_skip_after=party_list_skip_after,
            )
            if page_skip_after != skip_after:
                print(
                    f"using party-list skip-after {page_skip_after:.1f}s for {page.name}"
                )

            page_votes = extract_rows_with_retry(
                image_path=page,
                expected_max_ballot=len(rows),
                cache_dir=cache_dir,
                ocr_timeout=ocr_timeout,
                skip_after=page_skip_after,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_quota_waits=max_quota_waits,
                quota_buffer=quota_buffer,
            )
            for row in rows:
                vote = page_votes.get(row.ballot_number)
                if vote is None:
                    continue
                votes_by_id[row.id] = vote
            total_written = sum(1 for row in rows if votes_by_id.get(row.id, 0) > 0)
            write_submission(output_path, template_groups, votes_by_id, all_prefixes)
            write_missing_report(output_path, template_groups, votes_by_id, all_prefixes)
            if page_votes:
                completed_pages.add(page_key)
                write_progress(progress_path, completed_pages)
                if page_key in skipped_pages:
                    skipped_pages.remove(page_key)
                    rewrite_skipped_pages(skipped_path, skipped_pages)
            else:
                print(f"warning: parsed 0 rows from {page.name}; not marking completed")
                append_skipped_page(skipped_path, page_key, "parsed_0_rows_or_timeout")
                skipped_pages.add(page_key)
            print(
                f"updated after {page.name}: wrote {len(page_votes)} rows "
                f"for {prefix} ({total_written}/{len(rows)})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("submission_template.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("selected_images"))
    parser.add_argument("--all-images-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-csv", type=Path, default=Path("submission_typhoon.csv"))
    parser.add_argument("--api-key")
    parser.add_argument("--prefix", action="append", dest="prefixes")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument("--max-quota-waits", type=int, default=DEFAULT_MAX_QUOTA_WAITS)
    parser.add_argument("--quota-buffer", type=float, default=DEFAULT_QUOTA_BUFFER)
    parser.add_argument("--ocr-timeout", type=float, default=DEFAULT_OCR_TIMEOUT)
    parser.add_argument("--skip-after", type=float, default=DEFAULT_SKIP_AFTER)
    parser.add_argument("--party-list-skip-after", type=float, default=DEFAULT_PARTY_LIST_SKIP_AFTER)
    parser.add_argument("--start-from")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--retry-skipped-only", action="store_true")
    parser.add_argument("--retry-incomplete-prefixes", action="store_true")
    args = parser.parse_args()

    prefix_filter = set(args.prefixes) if args.prefixes else None
    run_incremental(
        template_path=args.template,
        images_dir=args.images_dir,
        output_path=args.output_csv,
        all_images_dir=args.all_images_dir,
        api_key=args.api_key,
        prefix_filter=prefix_filter,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_quota_waits=args.max_quota_waits,
        quota_buffer=args.quota_buffer,
        ocr_timeout=args.ocr_timeout,
        skip_after=args.skip_after,
        party_list_skip_after=args.party_list_skip_after,
        start_from=args.start_from,
        start_index=args.start_index,
        reset_progress=args.reset_progress,
        retry_skipped_only=args.retry_skipped_only,
        retry_incomplete_prefixes=args.retry_incomplete_prefixes,
    )


if __name__ == "__main__":
    main()
