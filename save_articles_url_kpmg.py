import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get('https://kpmg.com/xx/en/our-insights.html')
wait = WebDriverWait(driver, 10)
article_links = []

try:
    accept_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept All Cookies')]"))
    )
    accept_button.click()
    time.sleep(2)
except Exception as e:
    print("No cookie banner found or already accepted:", e)

def extract_links():
    """Extracts article links from the current page and appends new links."""
    container = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'cmp-filterlist__result-container')))
    links = container.find_elements(By.CLASS_NAME, 'cmp-filterlist__tile--action-link')
    for link in links:
        href = link.get_attribute('href')
        if href not in article_links:
            article_links.append(href)
            if len(article_links) >= 70:
                break

extract_links()

while len(article_links) < 70:
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        pagination_container = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.cmp-filterlist__pagination--numbers'))
        )
        
        driver.execute_script("arguments[0].scrollIntoView();", pagination_container)
        time.sleep(1)
        
        page_buttons = pagination_container.find_elements(By.TAG_NAME, 'button')
        
        active_index = None
        for i, btn in enumerate(page_buttons):
            if 'active' in btn.get_attribute('class'):
                active_index = i
                break
        
        if active_index is not None and active_index < len(page_buttons) - 1:
            next_page_button = page_buttons[active_index + 1]
            next_page_button.click()
            time.sleep(3)
            extract_links()
        else:
            print("No next page available.")
            break
    except Exception as e:
        print("Error during pagination:", e)
        break

article_links = article_links[:70]

print("Total articles collected:", len(article_links))
for link in article_links:
    print(link)

# Save the URLs to a CSV file
csv_filename = "article_links.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["url"])  # CSV header
    for link in article_links:
        writer.writerow([link])
print(f"Saved {len(article_links)} links to {csv_filename}")

driver.quit()
