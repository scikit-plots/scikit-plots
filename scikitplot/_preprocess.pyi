from collections.abc import Callable

def _preprocess_data(
    func: Callable | None = ...,
    *,
    replace_names: list[str] | None = ...,
    label_namer: str | None = ...,
) -> Callable: ...
