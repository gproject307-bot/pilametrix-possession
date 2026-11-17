"""Put the two upload files in submit/, named after the team, as the organisers ask.

    python finalize.py "<team name>"

Copies out/submission.csv to submit/<team>.csv and docs/technical_document.md to submit/<team>.md,
with the team name written into the document. If the `markdown` package and Chrome or Edge are
installed it also prints submit/<team>.pdf. Before copying it checks the CSV with the same rules
as the challenge's validate_submission.py: header, row count, frame order, labels.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
N_FRAMES = 199319
LABELS = {"H", "A", "D", "X"}
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def check_csv(path):
    """Raise ValueError unless the file passes the challenge's format checks."""
    with open(path, newline="") as handle:
        header = handle.readline().strip().lower().replace(" ", "")
        if header != "frame,label":
            raise ValueError("header is {!r}, expected frame,label".format(header))
        count = 0
        for expected, line in enumerate(handle):
            frame, label = line.rstrip("\r\n").split(",")
            if int(frame) != expected:
                raise ValueError("row {} has frame {}".format(expected, frame))
            if label.strip().upper() not in LABELS:
                raise ValueError("row {} has label {!r}".format(expected, label))
            count += 1
    if count != N_FRAMES:
        raise ValueError("{} rows, expected {}".format(count, N_FRAMES))


def safe_name(team):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", team).strip().rstrip(".")
    if not name:
        raise SystemExit("the team name has no characters a file name can use")
    return name


def write_pdf(markdown_path, pdf_path, title):
    """Markdown -> HTML -> PDF with a headless browser. Returns False when either is missing."""
    try:
        import markdown
    except ImportError:
        return False
    browser = next((path for path in BROWSERS if os.path.isfile(path)), None)
    if browser is None:
        return False
    with open(markdown_path, encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables", "fenced_code"])
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
@page {{ size: A4; margin: 18mm 16mm 20mm;
         @bottom-center {{ content: counter(page); font: 8pt "Segoe UI", sans-serif; color: #666; }} }}
body {{ font-family: "Segoe UI", system-ui, sans-serif; font-size: 10pt; line-height: 1.42;
       margin: 0; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 6pt; }} h2 {{ font-size: 13.5pt; margin: 18pt 0 6pt; }}
h3 {{ font-size: 11.5pt; margin: 14pt 0 4pt; }}
p {{ margin: 6pt 0; orphans: 3; widows: 3; }}
ul, ol {{ margin: 4pt 0 8pt; padding-left: 18pt; }}
table {{ border-collapse: collapse; margin: 8pt 0; font-size: 9pt; }}
th, td {{ border: 1px solid #bbb; padding: 3pt 6pt; text-align: left; vertical-align: top; }}
code, pre {{ font-family: Consolas, monospace; font-size: 9pt; }}
pre {{ background: #f4f4f4; padding: 6pt; white-space: pre-wrap; }}
/* page breaks: keep a heading or a lead-in line with what follows, and keep blocks whole */
h1, h2, h3, h2 + p, h3 + p, p:has(+ ul), p:has(+ ol), p:has(+ table), p:has(+ pre) {{
  break-after: avoid; }}
table, tr, pre, ul, ol, li {{ break-inside: avoid; }}
</style></head><body>{body}</body></html>""".format(title=title, body=body)
    with tempfile.TemporaryDirectory() as folder:
        html_path = os.path.join(folder, "document.html")
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(page)
        command = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
                   "--user-data-dir=" + os.path.join(folder, "profile"),
                   "--no-pdf-header-footer", "--print-to-pdf=" + os.path.abspath(pdf_path),
                   "file:///" + html_path.replace("\\", "/")]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    return os.path.isfile(pdf_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("team", help="the team name every member registered with")
    parser.add_argument("--csv", default=os.path.join(HERE, "out", "submission.csv"))
    parser.add_argument("--doc", default=os.path.join(HERE, "docs", "technical_document.md"))
    parser.add_argument("--out", default=os.path.join(HERE, "submit"))
    args = parser.parse_args()

    name = safe_name(args.team)
    try:
        check_csv(args.csv)
    except (OSError, ValueError) as error:
        raise SystemExit("{} is not a valid submission: {}".format(args.csv, error))
    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(args.out, name + ".csv")
    shutil.copyfile(args.csv, csv_path)

    with open(args.doc, encoding="utf-8") as handle:
        document = handle.read().replace("`<team_name>`", args.team)
    md_path = os.path.join(args.out, name + ".md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)

    pdf_path = os.path.join(args.out, name + ".pdf")
    made_pdf = write_pdf(md_path, pdf_path, name)

    with open(csv_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("ready to upload, in {}:".format(os.path.abspath(args.out)))
    print("  {}  (sha256 {})".format(os.path.basename(csv_path), digest))
    print("  {}".format(os.path.basename(md_path)))
    if made_pdf:
        print("  {}".format(os.path.basename(pdf_path)))
    else:
        print("  (no PDF: needs the markdown package and Chrome or Edge; the .md is accepted too)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
