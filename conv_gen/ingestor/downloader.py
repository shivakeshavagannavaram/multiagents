"""Download ToolBench data from the official Google Drive source."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLBENCH_GDRIVE_URL = "https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL"
TOOLBENCH_DATA_FILE_ID = "1ceLQ9S1IkFTiWeJ3G1FArsD4zY6WYiLa"


class ToolBenchDownloader:
    """Downloads and caches ToolBench tool definitions from Google Drive."""

    def __init__(self, cache_dir: str = ".cache/toolbench"):
        self.cache_dir = Path(cache_dir)
        self.tools_file = self.cache_dir / "tools.json"
        self._data_dir = self.cache_dir / "data"

    def is_cached(self) -> bool:
        if not self.tools_file.exists():
            return False
        try:
            with open(self.tools_file) as f:
                data = json.load(f)
            return len(data) > 0
        except Exception:
            return False

    def download(self, force: bool = False) -> Path:
        """Download, extract, and cache the full ToolBench tool catalogue (~10,600 tools)."""
        if self.is_cached() and not force:
            logger.info("ToolBench data already cached at %s", self.tools_file)
            return self.tools_file

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        tools_dir = self._data_dir / "toolenv" / "tools"
        if not tools_dir.exists():
            self._download_and_extract()

        tools_raw = self._load_all_tool_definitions(tools_dir)

        resp_dir = self._data_dir / "toolenv" / "response_examples"
        if resp_dir.exists():
            self._merge_response_schemas(tools_raw, resp_dir)

        logger.info("Loaded %d tool definitions (full dataset)", len(tools_raw))

        with open(self.tools_file, "w") as f:
            json.dump(tools_raw, f, indent=2)

        return self.tools_file

    def _download_and_extract(self) -> None:
        """Download data.zip from Google Drive and extract toolenv/ and instruction/."""
        try:
            import gdown
        except ImportError:
            raise ImportError(
                "gdown is required to download ToolBench data. "
                "Install it with: pip install gdown"
            )

        zip_path = self.cache_dir / "data.zip"

        if not zip_path.exists():
            logger.info("Downloading ToolBench data.zip from Google Drive...")
            gdown.download(
                id=TOOLBENCH_DATA_FILE_ID,
                output=str(zip_path),
                quiet=False,
            )

        logger.info("Extracting from data.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [
                m for m in zf.namelist()
                if m.startswith("data/toolenv/")
                and not m.startswith("__MACOSX")
            ]
            zf.extractall(self.cache_dir, members=members)

        logger.info("Extraction complete")

    def _load_all_tool_definitions(self, tools_dir: Path) -> list[dict]:
        """Load every tool JSON in the extracted ToolBench archive."""
        tools = []
        errors = 0

        for category_dir in sorted(tools_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for tool_file in sorted(category_dir.glob("*.json")):
                try:
                    with open(tool_file) as f:
                        raw = json.load(f)

                    if not isinstance(raw, dict):
                        continue
                    if "api_list" not in raw and "tool_name" not in raw:
                        continue

                    raw["category"] = category
                    raw["standardized_name"] = raw.get(
                        "standardized_name", tool_file.stem
                    )

                    tools.append(raw)

                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        logger.debug("Error loading %s: %s", tool_file, e)

        if errors > 0:
            logger.info("Encountered %d load errors (non-fatal)", errors)

        logger.info(
            "Loaded ALL %d tools across %d categories",
            len(tools), len({t["category"] for t in tools}),
        )
        return tools

    def _merge_response_schemas(
        self, tools: list[dict], resp_dir: Path
    ) -> None:
        """Fold response_examples schemas into each tool's api_list in place."""
        merged = 0

        tool_lookup: dict[tuple[str, str], dict] = {}
        for tool in tools:
            cat = tool.get("category", "")
            stem = tool.get("standardized_name", "")
            if cat and stem:
                tool_lookup[(cat, stem)] = tool

        for category_dir in resp_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for resp_file in category_dir.glob("*.json"):
                try:
                    with open(resp_file) as f:
                        resp_data = json.load(f)

                    api_list = resp_data.get("api_list", [])
                    if not api_list:
                        continue

                    tool = tool_lookup.get((category, resp_file.stem))
                    if not tool:
                        continue

                    api_by_name = {
                        api.get("name", ""): api
                        for api in tool.get("api_list", [])
                    }

                    for resp_api in api_list:
                        api_name = resp_api.get("name", "")
                        schema = resp_api.get("schema")
                        if api_name and schema and api_name in api_by_name:
                            existing = api_by_name[api_name]
                            if not existing.get("schema"):
                                existing["schema"] = schema
                                merged += 1

                except Exception as e:
                    logger.debug(
                        "Error loading response example %s: %s",
                        resp_file, e,
                    )

        if merged > 0:
            logger.info(
                "Merged %d response schemas from response examples", merged
            )
