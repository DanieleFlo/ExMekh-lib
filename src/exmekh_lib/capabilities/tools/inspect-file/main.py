"""Expose read-only workspace inspection through the standard tool runtime."""

from pathlib import Path

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.file_inspection import FILE_INSPECTION, FILE_INSPECTION_TEXT
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.tokenization import current_inspection_tokenization
from exmekh_lib.utils.tool_errors import missing_dependency, tool_error_result


@deep_tool(
    name=DEFAULT_CAPABILITIES.inspect_file,
    description=(FILE_INSPECTION_TEXT.inspect_description),
)
def run(
    work_root: Path,
    path: str,
    detailed: bool = False,
    max_items: int = FILE_INSPECTION.default_max_items,
    offset: int = 0,
    query: str | None = None,
    read_from_line: int | None = None,
) -> dict[str, object]:
    try:
        from exmekh_lib.core.file_inspection.service import inspect_file

        tokenization = current_inspection_tokenization()
        return inspect_file(
            work_root,
            path,
            tokenizer=tokenization.tokenizer,
            detailed=detailed,
            max_items=max_items,
            offset=offset,
            query=query,
            read_from_line=read_from_line,
            output_settings=tokenization.output_settings,
        )
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except FileInspectionError as error:
        return tool_error_result(error, "invalid_file_inspection")
