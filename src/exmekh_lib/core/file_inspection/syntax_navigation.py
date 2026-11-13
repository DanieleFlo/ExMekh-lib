"""Build searchable source regions from Tree-sitter syntax trees."""

import tomllib
from collections.abc import Callable
from html import unescape

import tree_sitter_c
import tree_sitter_css
import tree_sitter_html
import tree_sitter_javascript
import tree_sitter_toml
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT

from .models import Analysis

type SyntaxRegion = tuple[str, str, Node]

_JS_DECLARATIONS = {
    "abstract_class_declaration": "class",
    "class_declaration": "class",
    "class": "class",
    "enum_declaration": "enum",
    "function_declaration": "function",
    "function_expression": "function",
    "function_signature": "function",
    "generator_function_declaration": "function",
    "generator_function": "function",
    "interface_declaration": "interface",
    "internal_module": "namespace",
    "method_definition": "method",
    "type_alias_declaration": "type",
}
_JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "generator_function"}
_HTML_TITLES = {"title", "h1", "h2", "h3", "h4", "h5", "h6"}
_HTML_LANDMARKS = {
    "head",
    "body",
    "main",
    "header",
    "footer",
    "nav",
    "aside",
    "section",
    "article",
    "form",
    "table",
    "script",
    "style",
    "template",
}
_CSS_AT_RULES = {
    "charset_statement": "@charset",
    "import_statement": "@import",
    "namespace_statement": "@namespace",
    "media_statement": "@media",
    "supports_statement": "@supports",
    "keyframes_statement": "@keyframes",
}
_TOML_KEYS = {"bare_key", "quoted_key", "dotted_key"}


def _node_text(node: Node | None) -> str:
    return (node.text or b"").decode() if node is not None else ""


def _named_child(node: Node, index: int = 0) -> Node | None:
    return node.named_child(index) if index < node.named_child_count else None


def _javascript_region(node: Node) -> SyntaxRegion | None:
    parent = node.parent
    if node.type == "import_statement":
        return "import", _node_text(node.child_by_field_name("source")), node
    if (
        node.type == "arrow_function"
        and parent is not None
        and parent.type == "export_statement"
    ):
        return "function", "default", parent
    if node.type in {"field_definition", "public_field_definition"}:
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            field = "property" if node.type == "field_definition" else "name"
            return "method", _class_member_name(node, field), node
        return None
    if node.type == "variable_declarator":
        name = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name is None or name.type != "identifier" or value is None:
            return None
        if value.type not in _JS_FUNCTION_VALUES and value.type != "class":
            return None
        span = parent if parent is not None and parent.named_child_count == 1 else node
        exported = span.parent
        if exported is not None and exported.type == "export_statement":
            span = exported
        return "class" if value.type == "class" else "function", _node_text(name), span

    kind = _JS_DECLARATIONS.get(node.type)
    if kind is None or (parent is not None and parent.type == "variable_declarator"):
        return None
    name = _node_text(node.child_by_field_name("name"))
    if parent is not None and parent.type == "export_statement":
        return kind, name or "default", parent
    if kind == "method":
        name = _class_member_name(node, "name")
    return kind, name, node


def _class_member_name(node: Node, field: str) -> str:
    name = _node_text(node.child_by_field_name(field))
    body = node.parent
    if not name or body is None or body.type != "class_body":
        return name
    owner = body.parent
    if owner is None:
        return name
    name_node = owner.child_by_field_name("name")
    if (
        name_node is None
        and owner.parent is not None
        and owner.parent.type == "variable_declarator"
    ):
        name_node = owner.parent.child_by_field_name("name")
    owner_name = _node_text(name_node)
    return f"{owner_name}.{name}" if owner_name else name


def _html_region(node: Node) -> SyntaxRegion | None:
    if node.type not in {"element", "script_element", "style_element"}:
        return None
    tag = _named_child(node)
    if tag is None or tag.type not in {"start_tag", "self_closing_tag"}:
        return None
    name = _node_text(_named_child(tag)).lower()
    for attribute in tag.named_children:
        if attribute.type != "attribute":
            continue
        if _node_text(_named_child(attribute)).lower() == "id":
            identifier = _node_text(_named_child(attribute, 1)).strip("\"'")
            if identifier:
                name += f"#{identifier}"
            break
    tag_name = name.partition("#")[0]
    if tag_name in _HTML_TITLES:
        title = _html_heading_text(node)
        return "heading", f"{name} {title}" if title else name, node
    if name == tag_name and tag_name not in _HTML_LANDMARKS:
        return None
    return "element", name, node


def _html_heading_text(node: Node) -> str:
    parts: list[str] = []
    pending = list(reversed(node.named_children))
    while pending:
        child = pending.pop()
        if child.type in {"text", "entity"}:
            parts.append(unescape(_node_text(child)))
        elif child.type == "element":
            pending.extend(reversed(child.named_children))
    return " ".join(parts).strip()


def _css_region(node: Node) -> SyntaxRegion | None:
    if node.type == "rule_set":
        return "selector", _node_text(_named_child(node)), node
    if node.type == "at_rule":
        header: list[str] = []
        for child in node.named_children:
            if child.type == "block":
                break
            if child.type != "comment":
                header.append(_node_text(child))
        return "at_rule", " ".join(header), node
    keyword = _CSS_AT_RULES.get(node.type)
    if keyword is None:
        return None
    if node.type not in {"charset_statement", "namespace_statement"}:
        condition = _node_text(_named_child(node))
        keyword = f"{keyword} {condition}" if condition else keyword
    return "at_rule", keyword, node


def _c_region(node: Node) -> SyntaxRegion | None:
    if node.type == "preproc_include":
        return "include", _node_text(node.child_by_field_name("path")), node
    if node.type in {"preproc_def", "preproc_function_def"}:
        return "macro", _node_text(node.child_by_field_name("name")), node
    if node.type == "type_definition":
        return "typedef", _node_text(node.child_by_field_name("declarator")), node
    if (
        node.type in {"struct_specifier", "union_specifier", "enum_specifier"}
        and node.child_by_field_name("body") is not None
    ):
        return (
            node.type.removesuffix("_specifier"),
            _node_text(node.child_by_field_name("name")),
            node,
        )
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        identifier, _ = _c_declarator_name(declarator)
        return "function", identifier, node
    if node.type in {
        "function_declarator",
        "pointer_declarator",
        "parenthesized_declarator",
        "attributed_declarator",
    }:
        parent = node.parent
        if (
            parent is not None
            and parent.type == "declaration"
            and node in parent.children_by_field_name("declarator")
        ):
            identifier, wrapper = _c_declarator_name(node)
            if wrapper == "function_declarator":
                return "function", identifier, parent
    return None


def _c_declarator_name(declarator: Node | None) -> tuple[str, str]:
    innermost_wrapper = ""
    while declarator is not None and declarator.type != "identifier":
        if declarator.type == "parenthesized_declarator":
            declarator = _named_child(declarator)
        else:
            innermost_wrapper = declarator.type
            declarator = declarator.child_by_field_name("declarator")
    return _node_text(declarator), innermost_wrapper


class _TomlNavigation:
    """Track array-table occurrences while visiting the source in document order."""

    def __init__(self) -> None:
        self._array_counts: dict[str, int] = {}
        self._array_positions: dict[int, int] = {}

    def region(self, node: Node) -> SyntaxRegion | None:
        if node.type not in {"table", "table_array_element", "pair"}:
            return None
        label = self._label(node)
        if not label:
            return None
        if node.type != "pair":
            return "table", label, node
        parts = [label]
        parent = node.parent
        while parent is not None:
            if parent.type in {"table", "table_array_element", "pair"}:
                prefix = self._label(parent)
                if prefix:
                    parts.append(prefix)
            parent = parent.parent
        return "key", ".".join(reversed(parts)), node

    def _label(self, node: Node) -> str:
        key = next(
            (child for child in node.named_children if child.type in _TOML_KEYS), None
        )
        label = _node_text(key)
        if node.type != "table_array_element" or not label:
            return label
        position = self._array_positions.get(node.id)
        if position is None:
            position = self._array_counts.get(label, 0)
            self._array_counts[label] = position + 1
            self._array_positions[node.id] = position
        return f"{label}[{position}]"


_TOML_LANGUAGE = Language(tree_sitter_toml.language())
_SYNTAX_ANALYZERS: dict[str, tuple[Language, Callable[[Node], SyntaxRegion | None]]] = {
    "c": (Language(tree_sitter_c.language()), _c_region),
    "css": (Language(tree_sitter_css.language()), _css_region),
    "html": (Language(tree_sitter_html.language()), _html_region),
    "javascript": (Language(tree_sitter_javascript.language()), _javascript_region),
    "typescript": (
        Language(tree_sitter_typescript.language_typescript()),
        _javascript_region,
    ),
    "tsx": (Language(tree_sitter_typescript.language_tsx()), _javascript_region),
}


def analyze_syntax(text: str, result: Analysis) -> None:
    # Tree-sitter rows count LF; align them with read-file's physical lines.
    source = text.replace("\r\n", "\n").replace("\r", "\n").encode()
    if result.format == "toml":
        tomllib.loads(text)
        result.details["root_kind"] = "mapping"
        root = Parser(_TOML_LANGUAGE).parse(source).root_node
        _index_syntax(root, _TomlNavigation().region, result)
    else:
        language, find_region = _SYNTAX_ANALYZERS[result.format]
        _index_syntax(Parser(language).parse(source).root_node, find_region, result)


def _index_syntax(
    root: Node, find_region: Callable[[Node], SyntaxRegion | None], result: Analysis
) -> None:
    result.details["syntax_recovered"] = root.has_error
    if root.has_error:
        result.warnings.append(FILE_INSPECTION_TEXT.recovered_syntax_warning)

    settings = result.index.settings
    pending = [(root, 0)]
    visited = 0
    while pending and visited < settings.max_structure_nodes:
        node, depth = pending.pop()
        visited += 1
        region = find_region(node) if not node.is_missing else None
        if region is not None:
            kind, name, span = region
            if name:
                start = span.start_point.row + 1
                end = span.end_point.row + int(span.end_point.column != 0)
                if not result.index.add(
                    kind, " ".join(name.split()), start, max(start, end)
                ):
                    break
        sibling = node.next_named_sibling
        if sibling is not None:
            pending.append((sibling, depth))
        child = _named_child(node)
        if child is not None:
            if depth < settings.max_structure_depth:
                pending.append((child, depth + 1))
            else:
                result.index.mark_truncated()
    if pending:
        result.index.mark_truncated()
