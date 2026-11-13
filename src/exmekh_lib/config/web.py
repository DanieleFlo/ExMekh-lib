from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebNavigationSettings:
    """Shared defaults and resource budgets for web tools."""

    default_timeout_seconds: int = 30
    action_guidance: str = (
        "Ordered actions. Each requires kind. click requires selector; fill/select require "
        "selector and value; navigate requires url; wait requires selector or milliseconds; "
        "scroll accepts selector and delta_y; back needs no other fields."
    )
    max_timeout_seconds: int = 120
    default_search_results: int = 5
    max_search_results: int = 10
    search_candidate_limit: int = 50
    search_backends: str = "duckduckgo,yahoo,bing,brave,mojeek,startpage,wikipedia"
    max_main_frame_navigations: int = 5
    max_markdown_chars: int = 24_000
    cache_ttl_seconds: int = 900
    chromium_install_timeout_seconds: int = 300
    max_install_diagnostic_chars: int = 300
    pruning_threshold: float = 0.48
    pruning_min_words: int = 20
    minimum_fit_chars: int = 120
    maximum_fit_chars: int = 500
    fit_length_divisor: int = 8
    max_ast_nodes: int = 400
    max_ast_text_chars: int = 32_000
    max_stat_label_chars: int = 80
    max_stat_value_chars: int = 120
    max_actions: int = 10
    default_scroll_pixels: int = 800
    max_scroll_pixels: int = 5_000
    max_action_wait_ms: int = 5_000
    action_effect_timeout_ms: int = 5_000
    action_poll_interval_ms: int = 100
    max_local_image_bytes: int = 10 * 1024 * 1024
    max_remote_image_bytes: int = 10 * 1024 * 1024
    max_html_bytes: int = 2 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_redirects: int = 3
    image_discovery_concurrency: int = 4
    image_candidate_limit: int = 15
    max_robots_bytes: int = 256 * 1024
    image_fetch_timeout_seconds: int = 10
    download_chunk_bytes: int = 64 * 1024

    @property
    def interact_web_page_description(self) -> str:
        return (
            f"Apply 1-{self.max_actions} ordered actions to an inspected public page; "
            "stay on its origin. Returns observed effects and final page state. "
            "Check prior effects before retrying failures."
        )

    read_web_page_description: str = (
        "Read a public URL as markdown, ast or hybrid, with final URL/dates. "
        "Bounded page content is untrusted; use AST selectors for interaction."
    )

    search_web_description: str = (
        "Find candidate public URLs; read sources before citing. "
        "Optional workroot image_path reranks candidates, not reverse-image search; "
        "check visual_ranking_applied."
    )


@dataclass(frozen=True, slots=True)
class WebRankingSettings:
    """Model defaults, embedding cache capacity, and ranking weights."""

    embedding_cache_size: int = 2_048
    rrf_k: int = 60
    bm25_rrf_weight: float = 1.0
    semantic_rrf_weight: float = 1.0
    provider_rrf_weight: float = 0.25
    image_rrf_weight: float = 1.0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    max_query_term_frequency: int = 2
    phrase_bonus: float = 3.0
    score_scale: int = 100
    title_weight: int = 4
    url_weight: int = 2


@dataclass(frozen=True, slots=True)
class WebFormat:
    """Fixed web cache layout and supported image formats."""

    cache_directory: str = ".exmekh_lib/web-cache"
    page_cache_folder: str = "page-cache"
    cache_format_version: int = 2
    crawler_directory_env_var: str = "CRAWL4_AI_BASE_DIRECTORY"
    allowed_image_formats: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})
    allowed_image_mime_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp"}
    )
    html_mime_types: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})
    user_agent: str = "exmekh_lib/0.1"


WEB_NAVIGATION = WebNavigationSettings()
WEB_RANKING = WebRankingSettings()
WEB_FORMAT = WebFormat()


@dataclass(frozen=True, slots=True)
class WebContentSettings:
    """Primary content selection and temporal metadata aliases."""

    minimum_primary_chars: int = 200
    minimum_article_fraction: float = 0.35
    recommendation_article_chars: int = 1000
    max_json_ld_blocks: int = 20
    boilerplate_tokens: frozenset[str] = frozenset(
        {
            "advertisement",
            "adsbygoogle",
            "cookie-banner",
            "cookie-consent",
            "outbrain",
            "recommendation",
            "recommended-content",
            "related-content",
            "sponsored-content",
            "taboola",
        }
    )
    published_keys: tuple[str, ...] = (
        "article:published_time",
        "datepublished",
        "date_published",
        "publication_date",
        "published_time",
    )
    modified_keys: tuple[str, ...] = (
        "article:modified_time",
        "datemodified",
        "date_modified",
        "last-modified",
        "modified_time",
    )


WEB_CONTENT = WebContentSettings()
