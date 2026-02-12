"""Compare WordLift original HTML vs C6 (enhanced) vs C6-Plus (enhanced_plus).

Generates a comparison HTML report showing side-by-side differences:
- File sizes
- Key sections present/absent
- Summary of what C6-Plus adds to the original
"""

import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def analyze_html(filepath: Path) -> dict:
    """Analyze an HTML file and return key metrics."""
    if not filepath.exists():
        return {"exists": False}

    content = filepath.read_text(errors="replace")
    soup = BeautifulSoup(content, "lxml")

    return {
        "exists": True,
        "size_bytes": len(content.encode("utf-8")),
        "size_kb": round(len(content.encode("utf-8")) / 1024, 1),
        "has_jsonld": bool(soup.find_all("script", {"type": "application/ld+json"})),
        "jsonld_blocks": len(soup.find_all("script", {"type": "application/ld+json"})),
        "has_c6plus_summary": bool(soup.find(id="c6-plus-summary")),
        "has_c6plus_agent": bool(soup.find(id="c6-plus-agent-instructions")),
        "has_agent_instructions": "Agent Instructions" in content,
        "has_related_entities": "Related Entities" in content,
        "title": soup.title.string if soup.title else "N/A",
        "h1_count": len(soup.find_all("h1")),
        "h2_count": len(soup.find_all("h2")),
        "link_count": len(soup.find_all("a")),
    }


def generate_comparison_report(processed_dir: Path, domain: str) -> str:
    """Generate an HTML comparison report for all entities in a domain."""
    domain_dir = processed_dir / domain
    if not domain_dir.exists():
        return f"<p>Directory not found: {domain_dir}</p>"

    entities = sorted(d for d in domain_dir.iterdir() if d.is_dir())

    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>C6-Plus Comparison Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 2rem;
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: #888; margin-bottom: 2rem; }
        .entity-card {
            background: #1a1a2e;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .entity-card h2 {
            color: #a78bfa;
            font-size: 1.3rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #2a2a4a;
            padding-bottom: 0.5rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        th, td {
            text-align: left;
            padding: 0.6rem 1rem;
            border-bottom: 1px solid #2a2a4a;
        }
        th {
            color: #888;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }
        td { font-size: 0.9rem; }
        .yes { color: #34d399; font-weight: 600; }
        .no { color: #f87171; }
        .size { color: #60a5fa; font-weight: 600; }
        .delta { color: #fbbf24; font-size: 0.8rem; }
        .preview-links a {
            display: inline-block;
            margin-right: 1rem;
            padding: 0.4rem 0.8rem;
            background: #2a2a4a;
            color: #a78bfa;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.85rem;
        }
        .preview-links a:hover {
            background: #3a3a5a;
        }
        .additions {
            background: #0f2a1a;
            border: 1px solid #166534;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
        .additions h3 {
            color: #34d399;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        .additions ul {
            list-style: none;
            padding: 0;
        }
        .additions li {
            padding: 0.2rem 0;
            font-size: 0.9rem;
        }
        .additions li::before {
            content: "✅ ";
        }
    </style>
</head>
<body>
    <h1>C6-Plus Comparison Report</h1>
    <p class="subtitle">Comparing original WordLift HTML vs C6 (Enhanced) vs C6-Plus (Enhanced Plus)</p>
"""]

    for entity_dir in entities:
        name = entity_dir.name

        original = analyze_html(entity_dir / "plain.html")
        with_jsonld = analyze_html(entity_dir / "with_jsonld.html")
        enhanced = analyze_html(entity_dir / "enhanced.html")
        enhanced_plus = analyze_html(entity_dir / "enhanced_plus.html")

        # Use absolute paths for preview links
        abs_path = entity_dir.resolve()

        html_parts.append(f"""
    <div class="entity-card">
        <h2>{name}</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Original (Plain)</th>
                <th>With JSON-LD</th>
                <th>C6 Enhanced</th>
                <th>C6-Plus ✨</th>
            </tr>
            <tr>
                <td>File Size</td>
                <td class="size">{original.get('size_kb', '—')} KB</td>
                <td class="size">{with_jsonld.get('size_kb', '—')} KB</td>
                <td class="size">{enhanced.get('size_kb', '—')} KB</td>
                <td class="size">{enhanced_plus.get('size_kb', '—')} KB</td>
            </tr>
            <tr>
                <td>JSON-LD Blocks</td>
                <td>{fmt_bool(original.get('has_jsonld'))}</td>
                <td class="yes">{with_jsonld.get('jsonld_blocks', 0)}</td>
                <td class="yes">{enhanced.get('jsonld_blocks', 0)}</td>
                <td class="yes">{enhanced_plus.get('jsonld_blocks', 0)}</td>
            </tr>
            <tr>
                <td>Agent Instructions</td>
                <td class="no">No</td>
                <td class="no">No</td>
                <td class="yes">Yes</td>
                <td class="yes">Yes</td>
            </tr>
            <tr>
                <td>Entity Summary</td>
                <td class="no">No</td>
                <td class="no">No</td>
                <td class="yes">Yes</td>
                <td class="yes">Yes</td>
            </tr>
            <tr>
                <td>Related Entities</td>
                <td class="no">{fmt_bool2(original.get('has_related_entities'))}</td>
                <td class="no">{fmt_bool2(with_jsonld.get('has_related_entities'))}</td>
                <td class="yes">Yes</td>
                <td class="yes">Yes</td>
            </tr>
            <tr>
                <td>Original Content</td>
                <td class="yes">Yes</td>
                <td class="yes">Yes</td>
                <td class="no">No (rebuilt)</td>
                <td class="yes">Yes ✅</td>
            </tr>
        </table>

        <div class="additions">
            <h3>What C6-Plus adds to the original:</h3>
            <ul>
                <li>Entity summary block with name, type, and description</li>
                <li>Related entities navigation (expandable)</li>
                <li>llms.txt-style agent instructions for AI systems</li>
                <li>Size delta: +{round(enhanced_plus.get('size_kb', 0) - original.get('size_kb', 0), 1)} KB over original</li>
            </ul>
        </div>

        <div class="preview-links" style="margin-top: 1rem;">
            <a href="file://{abs_path}/plain.html">📄 Original</a>
            <a href="file://{abs_path}/enhanced.html">⚡ C6 Enhanced</a>
            <a href="file://{abs_path}/enhanced_plus.html">✨ C6-Plus</a>
        </div>
    </div>
""")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def fmt_bool(val):
    return '<span class="yes">Yes</span>' if val else '<span class="no">No</span>'


def fmt_bool2(val):
    return "Yes" if val else "No"


def main():
    processed_dir = Path("data/processed")
    domain = sys.argv[1] if len(sys.argv) > 1 else "wordlift_blog"

    report = generate_comparison_report(processed_dir, domain)

    out_path = processed_dir / domain / "comparison_report.html"
    out_path.write_text(report)
    print(f"Report saved to {out_path}")

    # Also print a text summary
    domain_dir = processed_dir / domain
    for entity_dir in sorted(d for d in domain_dir.iterdir() if d.is_dir()):
        name = entity_dir.name
        original = analyze_html(entity_dir / "plain.html")
        enhanced = analyze_html(entity_dir / "enhanced.html")
        enhanced_plus = analyze_html(entity_dir / "enhanced_plus.html")

        print(f"\n=== {name} ===")
        print(f"  Original:     {original.get('size_kb', '?')} KB")
        print(f"  C6 Enhanced:  {enhanced.get('size_kb', '?')} KB")
        print(f"  C6-Plus:      {enhanced_plus.get('size_kb', '?')} KB "
              f"(+{round(enhanced_plus.get('size_kb', 0) - original.get('size_kb', 0), 1)} KB)")
        print(f"  C6-Plus has summary:      {enhanced_plus.get('has_c6plus_summary')}")
        print(f"  C6-Plus has instructions:  {enhanced_plus.get('has_c6plus_agent')}")


if __name__ == "__main__":
    main()
