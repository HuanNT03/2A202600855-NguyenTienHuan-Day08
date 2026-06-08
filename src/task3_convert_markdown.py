"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path
import subprocess
import tempfile

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    failed_files = []

    if not legal_dir.exists():
        print(f"Directory {legal_dir} does not exist.")
        return failed_files

    for filepath in legal_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                if filepath.suffix.lower() == ".doc":
                    # Fallback conversion via libreoffice
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir_path = Path(temp_dir)
                        # Run libreoffice conversion to docx
                        cmd = [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "docx",
                            str(filepath),
                            "--outdir",
                            str(temp_dir_path)
                        ]
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if result.returncode != 0:
                            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr or result.stdout}")
                        
                        # Find the converted docx
                        converted_docx = temp_dir_path / f"{filepath.stem}.docx"
                        if not converted_docx.exists():
                            docx_files = list(temp_dir_path.glob("*.docx"))
                            if docx_files:
                                converted_docx = docx_files[0]
                            else:
                                raise FileNotFoundError("Converted DOCX file not found in temporary directory.")
                        
                        conversion_result = md.convert(str(converted_docx))
                else:
                    conversion_result = md.convert(str(filepath))

                text_content = conversion_result.text_content
                if not text_content or len(text_content.strip()) < 200:
                    warning_msg = (
                        f"# Scanned Document: {filepath.name}\n\n"
                        f"Warning: This document appears to be a scanned PDF, image-only document, or has very little text.\n"
                        f"Text could not be fully extracted directly using standard text extraction tools.\n"
                        f"Please refer to the original file {filepath.name} or run an OCR tool to parse its full content.\n"
                    )
                    text_content = warning_msg + "\n" + (text_content or "")

                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(text_content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path}")

            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  ✗ Failed to convert {filepath.name}: {err_msg}")
                failed_files.append((filepath.name, err_msg))

    return failed_files


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_files = []

    if not news_dir.exists():
        print(f"Directory {news_dir} does not exist.")
        return failed_files

    for filepath in news_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Thêm metadata header
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                content = header + data.get("content_markdown", "")
                output_path.write_text(content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path}")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  ✗ Failed to convert {filepath.name}: {err_msg}")
                failed_files.append((filepath.name, err_msg))
        elif filepath.is_file() and filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                md = MarkItDown()
                if filepath.suffix.lower() == ".doc":
                    # Fallback conversion via libreoffice
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir_path = Path(temp_dir)
                        cmd = [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "docx",
                            str(filepath),
                            "--outdir",
                            str(temp_dir_path)
                        ]
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if result.returncode != 0:
                            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr or result.stdout}")
                        converted_docx = temp_dir_path / f"{filepath.stem}.docx"
                        if not converted_docx.exists():
                            docx_files = list(temp_dir_path.glob("*.docx"))
                            if docx_files:
                                converted_docx = docx_files[0]
                            else:
                                raise FileNotFoundError("Converted DOCX file not found in temporary directory.")
                        conversion_result = md.convert(str(converted_docx))
                else:
                    conversion_result = md.convert(str(filepath))

                text_content = conversion_result.text_content
                if not text_content or len(text_content.strip()) < 200:
                    warning_msg = (
                        f"# Scanned Document: {filepath.name}\n\n"
                        f"Warning: This document appears to be a scanned PDF, image-only document, or has very little text.\n"
                        f"Text could not be fully extracted directly using standard text extraction tools.\n"
                        f"Please refer to the original file {filepath.name} or run an OCR tool to parse its full content.\n"
                    )
                    text_content = warning_msg + "\n" + (text_content or "")

                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(text_content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path}")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  ✗ Failed to convert {filepath.name}: {err_msg}")
                failed_files.append((filepath.name, err_msg))

    return failed_files


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    failed_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    failed_news = convert_news_articles()

    print("\n" + "=" * 50)
    print("✓ Done! Output tại:", OUTPUT_DIR)

    all_failed = failed_legal + failed_news
    if all_failed:
        print("\n" + "!" * 50)
        print("WARNING: The following files could not be converted by MarkItDown:")
        for name, err in all_failed:
            print(f"  - {name}: {err}")
        print("!" * 50 + "\n")
    else:
        print("All documents converted successfully!")
    print("=" * 50)


if __name__ == "__main__":
    convert_all()
