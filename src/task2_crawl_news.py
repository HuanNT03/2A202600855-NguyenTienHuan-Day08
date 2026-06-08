"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_markdown_content(text: str) -> str:
    """
    Loại bỏ các tag HTML, các liên kết không liên quan hoặc thặng dư,
    và chỉ giữ lại phần chữ/tiêu đề của các đường link nội dung.
    """
    if not text:
        return ""
    
    # 1. Loại bỏ các ảnh markdown ![]() hoàn toàn
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', '', text)
    
    # 2. Chuyển đổi các markdown links [tiêu đề link](url) thành tiêu đề link
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    
    # 3. Loại bỏ các thẻ HTML còn sót lại
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Loại bỏ các đường link raw bắt đầu bằng http/https
    text = re.sub(r'https?://\S+', '', text)
    
    # 5. Các cụm từ cần bỏ qua (social share, copy links, navigation, footer links)
    ignore_patterns = [
        r"chia sẻ bài viết lên",
        r"sao chép liên kết",
        r"lưu bài viết",
        r"đăng nhập để",
        r"tải ứng dụng",
        r"độc giả gửi bài",
        r"tuyển dụng",
        r"liên hệ quảng cáo",
        r"báo giá",
        r"hỗ trợ kỹ thuật",
        r"cơ quan chủ quản",
        r"giấy phép số",
        r"tổng biên tập",
        r"tất cả quyền lợi",
        r"cấm sao chép",
        r"tin cùng chuyên mục",
        r"tin mới",
        r"xem thêm về",
        r"xem các bài viết",
        r"theo dõi trên",
        r"bình luận",
        r"copy link thành công",
        r"hotline:",
        r"email:",
        r"địa chỉ:",
        r"điện thoại:",
        r"all rights reserved",
        r"chỉ được phát hành lại"
    ]
    
    # Các từ khóa điều hướng/menu đặc trưng trên một số báo cần lọc bỏ hoàn toàn nếu đứng riêng lẻ
    menu_terms = {
        "thời sự", "chính trị", "kinh doanh", "dân tộc và tôn giáo", "giáo dục",
        "thế giới", "thể thao", "văn hóa - giải trí", "đời sống", "sức khỏe",
        "công nghệ", "xe", "bất động sản", "du lịch", "bạn đọc", "tin nóng",
        "tuần việt nam", "công nghiệp hỗ trợ", "giảm nghèo bền vững", "nông thôn mới",
        "dân tộc thiểu số và miền núi", "nội dung chuyên đề", "english", "hồ sơ",
        "ảnh", "video", "multimedia", "podcast", "24h qua", "tuyến bài", "sự kiện",
        "sự kiện nóng", "liên hệ tòa soạn", "lịch vạn niên", "độc giả gửi bài",
        "tuyển dụng", "tải ứng dụng", "aa", "bình luận", "phản hồi", "in bài viết",
        "trở lại thời sự", "trang chủ", "góc nhìn", "vĩ mô", "doanh nghiệp",
        "chứng khoán", "bất động sản", "quốc tế", "khoa học", "học đường",
        "tuyển sinh", "du học", "đối thoại", "nhạc", "phim", "thư giãn",
        "tâm sự", "thời trang", "làm đẹp", "nội trợ", "du lịch", "ẩm thực",
        "khỏe đẹp", "y học cổ truyền", "giới tính", "nhịp sống số", "thiết bị",
        "bảo mật", "xe điện", "xe máy", "diễn đàn", "nhà đất", "thị trường",
        "bản đồ", "dự án", "không gian sống", "điểm đến", "ẩm thực",
        "cẩm nang", "tour", "bạn đọc", "bảo vệ người tiêu dùng", "đường dây nóng",
        "hỏi đáp pháp luật", "tin xem nhiều", "tin mới nhất", "nóng 24h",
        "văn hóa", "giải trí", "talks"
    }

    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        should_ignore = False
        
        for pattern in ignore_patterns:
            if re.search(pattern, lower_line):
                should_ignore = True
                break
        if should_ignore:
            continue
            
        # Kiểm tra nếu dòng chỉ là menu hoặc mục lục nhỏ lẻ
        line_clean = re.sub(r'^[*#\-\s•+|]+', '', line).strip().lower()
        if line_clean in menu_terms:
            continue
            
        # Dọn các ký tự rác chỉ có định dạng
        line = re.sub(r'^[;*\s()|#\-+]+$', '', line).strip()
        line = line.strip(';').strip()
        
        if not line:
            continue
            
        # Bỏ qua các chỉ số hoặc dòng rác siêu ngắn
        if len(line) < 3 and line in [";", "*", "-", "|", "#", "1", "2", "3", "4", "1/4"]:
            continue
            
        cleaned_lines.append(line)
        
    # Nối các dòng lại, giữ một dòng trống giữa các đoạn văn cho đúng markdown
    result_lines = []
    for line in cleaned_lines:
        result_lines.append(line)
        result_lines.append("")
        
    return '\n'.join(result_lines).strip()


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vietnamnet.vn/vai-tro-ca-si-chi-dan-nguoi-mau-an-tay-trong-vu-4-tiep-vien-xach-ma-tuy-2502809.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # 1. Thử sử dụng requests + BeautifulSoup + markdownify để trích xuất nội dung chính xác và sạch nhất
    try:
        import requests
        from bs4 import BeautifulSoup
        import markdownify

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, headers=headers, timeout=15)
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        html_content = response.text
        
        soup = BeautifulSoup(html_content, "lxml")
        
        # Lấy tiêu đề bài báo
        title = "Unknown Title"
        meta_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()
        else:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text().strip()
            elif soup.title:
                title = soup.title.get_text().strip()
                
        # Loại bỏ các thẻ rác trước khi chuyển sang markdown
        for tag in soup(["script", "style", "nav", "footer", "iframe", "header"]):
            tag.decompose()
            
        # Tìm phần nội dung bài báo dựa trên các selector chính xác của từng trang
        content_soup = None
        for selector in [
            "div.fck_detail",          # VnExpress
            "article.fck_detail",      # VnExpress
            "div.vov-content",          # VOV
            "div.content-detail",       # VietnamNet
            "div.maincontent",          # VietnamNet
            "div.main-content",
            "div.article_detail",
            "div.content-wrapper", 
            "div.article-content",
            "article"                   # Thẻ article chung
        ]:
            found = soup.select_one(selector)
            if found:
                content_soup = found
                break
        
        if not content_soup:
            content_soup = soup.body or soup

        content_markdown = markdownify.markdownify(str(content_soup), heading_style="ATX").strip()
        cleaned_content = clean_markdown_content(content_markdown)
        
        if cleaned_content and len(cleaned_content) > 200:
            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": cleaned_content
            }
    except Exception as e:
        print(f"  ⚠ HTTP/BS4 crawler không khả dụng hoặc lỗi ({str(e)}). Chuyển sang Crawl4AI...")

    # 2. Fallback: Sử dụng Crawl4AI
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            content_markdown = result.markdown if result.markdown else ""
            
            title = "Unknown Title"
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                title = result.metadata.get('title', "Unknown Title")
            
            if title == "Unknown Title" and content_markdown:
                lines = [line.strip() for line in content_markdown.split('\n') if line.strip()]
                if lines:
                    title = lines[0].replace('#', '').strip()
            
            cleaned_content = clean_markdown_content(content_markdown)
            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": cleaned_content
            }
    except Exception as fallback_e:
        print(f"  ❌ Lỗi khi Crawl4AI crawl {url}: {str(fallback_e)}")
        return {
            "url": url,
            "title": "Error Processing",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": ""
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
