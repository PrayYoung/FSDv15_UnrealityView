"""Safe extensible identifiers shared by teacher caches and adapter artifacts."""
import re


def validate_style_slug(style):
    if not isinstance(style,str) or re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,31}',style) is None:
        raise ValueError('Style must be a lowercase safe slug of at most 32 characters')
    return style
