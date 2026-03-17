"""
Convert the unified whitepaper markdown to a styled HTML document.
Usage: python generate_html_whitepaper.py
"""
import os
import markdown

MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cascading-hallucinations-whitepaper.md")
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cascading-hallucinations-whitepaper.html")

with open(MD_PATH, "r") as f:
    md_text = f.read()

body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cascading Hallucinations in Self-Improving AI Systems</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    background: #fafaf9; color: #1c1917;
    max-width: 900px; margin: 0 auto; padding: 3rem 2rem;
    line-height: 1.75; font-size: 16px;
  }}
  h1 {{ font-size: 1.85rem; margin: 2rem 0 0.5rem; color: #0c0a09; line-height: 1.3; }}
  h2 {{ font-size: 1.4rem; margin: 2.5rem 0 0.75rem; color: #1c1917;
        border-bottom: 2px solid #d6d3d1; padding-bottom: 0.4rem; }}
  h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; color: #292524; }}
  h4 {{ font-size: 1rem; margin: 1.25rem 0 0.5rem; color: #44403c; }}
  p {{ margin: 0.75rem 0; }}
  strong {{ color: #0c0a09; }}
  code {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #f5f5f4; padding: 0.15em 0.4em; border-radius: 3px;
    font-size: 0.9em; color: #dc2626;
  }}
  pre {{
    background: #1c1917; color: #d6d3d1; padding: 1.25rem;
    border-radius: 8px; overflow-x: auto; margin: 1rem 0;
    font-size: 0.82rem; line-height: 1.5;
  }}
  pre code {{
    background: none; color: inherit; padding: 0; font-size: inherit;
  }}
  table {{
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th {{
    background: #292524; color: #fafaf9; padding: 0.6rem 0.75rem;
    text-align: left; font-weight: 600;
  }}
  td {{
    padding: 0.5rem 0.75rem; border-bottom: 1px solid #e7e5e4;
  }}
  tr:nth-child(even) {{ background: #f5f5f4; }}
  hr {{ border: none; border-top: 1px solid #d6d3d1; margin: 2rem 0; }}
  ul, ol {{ margin: 0.75rem 0 0.75rem 1.5rem; }}
  li {{ margin: 0.3rem 0; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  blockquote {{
    border-left: 4px solid #a8a29e; margin: 1rem 0; padding: 0.5rem 1rem;
    background: #f5f5f4; color: #57534e;
  }}
  .header-meta {{ color: #78716c; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  @media print {{
    body {{ max-width: none; padding: 1rem; font-size: 11pt; }}
    pre {{ font-size: 8pt; }}
    h2 {{ page-break-before: auto; }}
  }}
  @media (max-width: 640px) {{
    body {{ padding: 1rem; font-size: 15px; }}
    pre {{ font-size: 0.75rem; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""

with open(HTML_PATH, "w") as f:
    f.write(html)

print(f"Generated: {HTML_PATH}")
print(f"Size: {os.path.getsize(HTML_PATH):,} bytes")
