from typing import Any
import importlib

def resolve_class(class_name: str) -> Any:
    sections = class_name.split(".")
    m = importlib.import_module(".".join(sections[:-1]))
    c = getattr(m, sections[-1])
    return c
