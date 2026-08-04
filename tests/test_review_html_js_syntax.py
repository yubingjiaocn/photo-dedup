"""Validate embedded JavaScript in review.html for syntax errors.

The review page template contains embedded JS that must be syntactically valid.
This gate extracts the inline <script> blocks and validates them with node --check.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from src import review_page


def test_embedded_js_is_syntactically_valid():
    """Extract embedded JS from render_html and validate with node --check."""
    # Render minimal HTML to extract the script
    data = {
        "groups": [],
        "queues": {"MAYBE": [], "UNKNOWN": []},
        "delete_paths": [],
        "total_delete_bytes": 0,
    }
    html = review_page.render_html(data, Path("/tmp"), review_limit=0)

    # Extract all <script>...</script> blocks
    script_pattern = re.compile(r'<script>(.*?)</script>', re.DOTALL)
    scripts = script_pattern.findall(html)

    assert len(scripts) > 0, "No <script> blocks found in rendered HTML"

    # Combine all scripts and validate with node --check
    combined_js = "\n".join(scripts)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(combined_js)
        temp_path = Path(f.name)

    try:
        result = subprocess.run(
            ['node', '--check', str(temp_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            error_msg = (
                f"JavaScript syntax validation failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}\n"
                f"Extracted JS length: {len(combined_js)} chars"
            )
            raise AssertionError(error_msg)
    finally:
        temp_path.unlink(missing_ok=True)
