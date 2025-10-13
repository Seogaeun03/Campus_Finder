import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

CHROME_DRIVER_PATH = "./chromedriver.exe"
URLS = [f"https://www.donga.ac.kr/kor/CMS/Contents/Contents.do?mCode=MN{code}" for code in range(115, 170)]
SAVE_FOLDER = "Result_crawling"
os.makedirs(SAVE_FOLDER, exist_ok=True)

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

for url in URLS:
    driver.get(url)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = soup.find("div", id="contents")
    if not content_div:
        continue

    mcode = url.split("mCode=MN")[-1]
    result_text = f"[URL] {url}\n"

    last_was_title = False
    table_text_set = set()  # 표에서 수집한 모든 문장 저장용

    for elem in content_div.descendants:
        if elem.name in ["h2", "h3", "h4"]:
            title = elem.get_text(strip=True)
            if title:
                result_text += f"\n[제목] {title}\n"
                last_was_title = True

        elif elem.name in ["p", "li"]:
            text = elem.get_text(strip=True)
            if text:
                # 표에 이미 등장한 내용이면 생략
                if any(text in t or t in text for t in table_text_set):
                    continue

                # 제목 바로 뒤의 첫 문단만 [본문] 표시
                if last_was_title:
                    result_text += f"[본문] {text}\n"
                    last_was_title = False
                else:
                    result_text += f"{text}\n"

        elif elem.name == "table":
            rows = elem.find_all("tr")
            if rows:
                result_text += "[표 데이터]\n"
                for row in rows:
                    cells = []
                    for cell in row.find_all(["th", "td"]):
                        # 여러 줄 텍스트를 콤마로 병합
                        cell_text = ", ".join(cell.stripped_strings)
                        if cell_text:
                            cells.append(cell_text)
                            # 표 내용 중복 방지용으로 저장
                            table_text_set.add(cell_text)
                    if cells:
                        line = " | ".join(cells)
                        result_text += line + "\n"
                        table_text_set.add(line)

    file_path = os.path.join(SAVE_FOLDER, f"MN{mcode}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"✅ 저장 완료: MN{mcode}.txt")

driver.quit()
print("\n🎉 전체 크롤링 완료! 표 중복 제거 + 순서 보존 + RAG 구조 최적화 완료.")
