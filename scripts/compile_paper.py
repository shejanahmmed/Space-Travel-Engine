"""Academic Paper Compiler for Relativistic Space-Travel Computational Engine.

Compiles paper.md and paper.bib into a journal-formatted PDF using Microsoft Edge headless.
"""

from pathlib import Path
import re
import subprocess
import markdown

ROOT = Path(__file__).resolve().parent.parent
PAPER_MD = ROOT / "paper.md"
PAPER_BIB = ROOT / "paper.bib"
OUTPUT_HTML = ROOT / "paper.html"
OUTPUT_PDF = ROOT / "paper.pdf"

EDGE_PATH = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def parse_bibtex(bib_path: Path) -> dict[str, dict[str, str]]:
    """Parse simple bibtex file into key -> dict mapping."""
    content = bib_path.read_text(encoding="utf-8")
    entries = {}
    
    # Match @type{key, ...}
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,]+),([^@]*)\}", re.DOTALL)
    for match in pattern.finditer(content):
        entry_type, key, body = match.groups()
        key = key.strip()
        fields = {}
        for line in body.splitlines():
            field_match = re.match(r"\s*(\w+)\s*=\s*[\{\"](.*)[\}\"],?", line)
            if field_match:
                fname, fval = field_match.groups()
                fields[fname.lower()] = fval.strip().rstrip("},\"")
        entries[key] = {
            "type": entry_type.lower(),
            "fields": fields,
        }
    return entries


def format_reference(key: str, entry: dict[str, str]) -> str:
    """Format a single BibTeX entry into standard academic APA/IEEE citation."""
    f = entry["fields"]
    author = f.get("author", "Unknown Author")
    title = f.get("title", "Untitled")
    journal = f.get("journal", f.get("booktitle", f.get("institution", "")))
    volume = f.get("volume", "")
    pages = f.get("pages", "")
    year = f.get("year", "")
    doi = f.get("doi", "")
    url = f.get("url", f"https://doi.org/{doi}" if doi else "")

    ref_str = f"**{author}** ({year}). *{title}*."
    if journal:
        ref_str += f" _{journal}_"
    if volume:
        ref_str += f", {volume}"
    if pages:
        ref_str += f", pp. {pages}"
    ref_str += "."
    if url:
        ref_str += f" [[DOI/Link]({url})]"
    return ref_str


def main():
    bib_entries = parse_bibtex(PAPER_BIB)
    md_text = PAPER_MD.read_text(encoding="utf-8")

    # Extract YAML frontmatter
    yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, re.DOTALL)
    title = "Deterministic Relativistic Space-Travel Computational Engine"
    authors = ["Farjan Ahmmed"]
    date_str = "September 2026"

    body_md = md_text[yaml_match.end():] if yaml_match else md_text

    # Extract all citation keys in appearance order
    citation_keys = []
    def replace_citation(match):
        raw = match.group(1)
        keys = [k.strip().lstrip("@") for k in raw.split(";")]
        indices = []
        for k in keys:
            if k not in citation_keys:
                citation_keys.append(k)
            idx = citation_keys.index(k) + 1
            indices.append(f"[{idx}](#ref-{k})")
        return "<sup>" + ",".join(indices) + "</sup>"

    # Match [@key1; @key2] or [@key] or @key
    body_md = re.sub(r"\[@([^\]]+)\]", replace_citation, body_md)
    body_md = re.sub(r"(?<!\w)@([a-zA-Z0-9_-]+)", lambda m: replace_citation(re.match(r"(.*)", m.group(1))), body_md)

    # Format References list
    ref_md_lines = ["\n"]
    for idx, key in enumerate(citation_keys, start=1):
        if key in bib_entries:
            formatted = format_reference(key, bib_entries[key])
            ref_md_lines.append(f"<div id=\"ref-{key}\" class=\"reference-entry\"><span class=\"ref-num\">[{idx}]</span> {formatted}</div>\n")
        else:
            ref_md_lines.append(f"<div id=\"ref-{key}\" class=\"reference-entry\"><span class=\"ref-num\">[{idx}]</span> <code>{key}</code></div>\n")

    # Inject into # References section
    if "# References" in body_md:
        body_md = body_md.replace("# References", "# References\n" + "\n".join(ref_md_lines))
    else:
        body_md += "\n\n# References\n" + "\n".join(ref_md_lines)

    html_content = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "def_list", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=JetBrains+Mono:wght@400;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
  
  <!-- MathJax for high-fidelity LaTeX rendering in PDF -->
  <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>

  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 20mm 18mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}
    body {{
      font-family: 'Newsreader', Georgia, serif;
      font-size: 11pt;
      line-height: 1.55;
      color: #1a1a1a;
      background: #ffffff;
      max-width: 800px;
      margin: 0 auto;
      padding: 10px;
    }}
    h1.paper-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 22pt;
      font-weight: 700;
      line-height: 1.25;
      color: #0b192c;
      text-align: center;
      margin-top: 10px;
      margin-bottom: 12px;
    }}
    .author-block {{
      text-align: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1.5px solid #000;
    }}
    .author-name {{
      font-family: 'Outfit', sans-serif;
      font-size: 13pt;
      font-weight: 600;
      color: #1e3a8a;
    }}
    .author-affiliation {{
      font-size: 9.5pt;
      color: #4b5563;
      margin-top: 4px;
    }}
    .paper-meta {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 8.5pt;
      color: #6b7280;
      margin-top: 6px;
    }}
    h1 {{
      font-family: 'Outfit', sans-serif;
      font-size: 14pt;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 24px;
      margin-bottom: 10px;
      page-break-after: avoid;
    }}
    h2 {{
      font-family: 'Outfit', sans-serif;
      font-size: 12pt;
      font-weight: 600;
      color: #1e293b;
      margin-top: 18px;
      margin-bottom: 8px;
      page-break-after: avoid;
    }}
    h3 {{
      font-size: 11pt;
      font-weight: 600;
      margin-top: 14px;
      margin-bottom: 6px;
      page-break-after: avoid;
    }}
    p {{
      margin-bottom: 10px;
      text-align: justify;
      hyphens: auto;
    }}
    code {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 9.5pt;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 3px;
      color: #0f172a;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px 14px;
      overflow-x: auto;
      font-size: 9pt;
      margin: 12px 0;
      page-break-inside: avoid;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 9.5pt;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 6px 10px;
      text-align: left;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 600;
      font-family: 'Outfit', sans-serif;
    }}
    sup {{
      font-size: 8pt;
      font-weight: 600;
      color: #2563eb;
    }}
    sup a {{
      color: #2563eb;
      text-decoration: none;
    }}
    .reference-entry {{
      font-size: 9pt;
      line-height: 1.45;
      margin-bottom: 8px;
      padding-left: 28px;
      text-indent: -28px;
      text-align: left;
      page-break-inside: avoid;
    }}
    .ref-num {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: #1e3a8a;
      margin-right: 4px;
    }}
    .reference-entry a {{
      color: #0284c7;
      text-decoration: none;
      font-family: 'JetBrains Mono', monospace;
      font-size: 8pt;
    }}
  </style>
</head>
<body>
  <div class="header-container">
    <h1 class="paper-title">{title}</h1>
    <div class="author-block">
      <div class="author-name">Farjan Ahmmed</div>
      <div class="author-affiliation">Independent Research • Dhaka, Bangladesh • <a href="mailto:farjan.swe@gmail.com">farjan.swe@gmail.com</a></div>
      <div class="paper-meta">JOSS Submission • DE440 BCRS • 1PN/2PN General Relativity • {date_str}</div>
    </div>
  </div>

  {html_content}

</body>
</html>
"""

    OUTPUT_HTML.write_text(full_html, encoding="utf-8")
    print(f"Generated formatted paper HTML at: {OUTPUT_HTML}")

    # Compile to PDF using Microsoft Edge headless
    if EDGE_PATH.exists():
        print(f"Compiling PDF via Microsoft Edge headless...")
        cmd = [
            str(EDGE_PATH),
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={OUTPUT_PDF}",
            "--no-pdf-header-footer",
            str(OUTPUT_HTML.resolve().as_uri()),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and OUTPUT_PDF.exists():
            print(f"Successfully generated PDF: {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"Edge compilation finished with returncode {res.returncode}: {res.stderr}")
    else:
        print("Microsoft Edge not found at expected path.")


if __name__ == "__main__":
    main()
