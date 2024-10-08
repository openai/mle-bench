import json
from pathlib import Path

from playwright._impl._errors import TimeoutError
from playwright.sync_api import sync_playwright
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type(TimeoutError)),
)
def extract_text_from_div(page, url):
    page.goto(url)
    page.wait_for_selector('div[data-testid="discussions-topic-header"]')
    # Extract the text content from the div
    div_text = page.locator('div[data-testid="discussions-topic-header"]').inner_text()
    return div_text


if __name__ == "__main__":
    discussions_outdir = Path("discussions")
    comps_to_urls_path = Path("comps_to_urls.json")
    with open(comps_to_urls_path) as f:
        comps_to_urls = json.load(f)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for comp_slug, urls in tqdm(comps_to_urls.items()):
            for url in urls:
                outfile = discussions_outdir / comp_slug / f"{Path(url).stem}.txt"
                if outfile.exists():
                    continue  # No need to scrape again
                text = extract_text_from_div(page, url)
                outfile.parent.mkdir(parents=True, exist_ok=True)
                with open(outfile, "w") as f:
                    f.write(text)

        # Close the browser
        browser.close()
