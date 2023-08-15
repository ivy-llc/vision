# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"][
    "json_url"
] = "https://unify.ai/docs/versions/vision.json"

repo_name = "vision"
