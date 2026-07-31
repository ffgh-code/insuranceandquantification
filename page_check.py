from playwright.sync_api import sync_playwright
import time

pages_to_test = [
    "Overview", "Market Data", "Sentiment Analysis", "Volatility Models",
    "Strategy Backtest", "Actuarial Applications", "Rolling Backtest Report",
    "Regime Performance", "Solvency Simulation", "Reserving Comparison",
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    for pg in pages_to_test:
        try:
            page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
            time.sleep(2)
            link = page.query_selector(f"text={pg}")
            if not link:
                errors.append(f"{pg}: link not found")
                continue
            link.click()
            time.sleep(3)
            exc = page.query_selector('[data-testid="stException"]')
            if exc:
                errors.append(f"{pg}: EXCEPTION - {exc.inner_text()[:200]}")
            else:
                print(f"OK: {pg}")
        except Exception as e:
            errors.append(f"{pg}: EXCEPTION - {str(e)[:100]}")
    browser.close()
    print()
    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print(f"  {e}")
    else:
        print("ALL 10 PAGES OK")
