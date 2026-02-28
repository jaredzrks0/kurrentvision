import os
import argparse
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import random
import time
import tqdm
from typing import List


NAMESPACES = {
    "mets": "http://www.loc.gov/METS/",
    "xlink": "http://www.w3.org/1999/xlink",
}


def parse_mets(mets_path: str) -> List[str]:
    """
    Return ordered list of image URLs from METS
    """

    tree = ET.parse(mets_path)
    root = tree.getroot()

    file_map = {}

    for file_el in root.findall(".//mets:file", NAMESPACES):
        file_id = file_el.attrib.get("ID")
        flocat = file_el.find(".//mets:FLocat", NAMESPACES)

        if file_id and flocat is not None:
            href = flocat.attrib.get(
                "{http://www.w3.org/1999/xlink}href"
            )
            if href:
                file_map[file_id] = href

    pages = []

    # Get pages in correct order
    for div in root.findall(".//mets:div[@TYPE='page']", NAMESPACES):
        order = div.attrib.get("ORDER")
        fptr = div.find(".//mets:fptr", NAMESPACES)

        if order and fptr is not None:
            file_id = fptr.attrib.get("FILEID")
            url = file_map.get(file_id)

            if url:
                pages.append((int(order), url))

    pages.sort(key=lambda x: x[0])

    return [url for _, url in pages]


def get_extension(url: str) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext if ext else ".jpg"


def download_all(mets_path: str, output_dir: str, viewer_url: str) -> None:
    urls = parse_mets(mets_path)

    print(f"Found {len(urls)} images")

    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Wait for page okay
        page.goto(viewer_url)
        page.wait_for_timeout(6000)

        # Download images
        for (i, url) in tqdm.tqdm(enumerate(urls, start=1)):
            # Give the page a break
            time.sleep(random.uniform(2, 6))
            if random.random() < 0.15:
                time.sleep(random.uniform(5, 12))
            
            ext = get_extension(url)
            filename = f"{i:08d}{ext}"
            filepath = os.path.join(output_dir, filename)

            print(f"{i}/{len(urls)} -> {filename}")

            try:
                # Go to the url of the given image
                response = page.goto(url)
                data = response.body()

                # Download the image
                with open(filepath, "wb") as f:
                    f.write(data)

            except Exception as e:
                print(f"Failed: {url}")
                print(e)

        browser.close()

    print("Done.")

### RUNNER ###

def main():
    parser = argparse.ArgumentParser(
        description="Download images from METS using Playwright"
    )

    parser.add_argument(
        "-m",
        "--mets",
        required=True,
        help="Relative or absolute path to METS XML file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="images",
        help="Output directory (default: images)",
    )

    parser.add_argument(
        "-v",
        "--viewer",
        default="https://www.digitale-bibliothek-mv.de/",
        help="Viewer URL to solve anti-bot challenge",
    )

    args = parser.parse_args()

    mets_path = os.path.abspath(args.mets)

    if not os.path.exists(mets_path):
        raise FileNotFoundError(f"METS file not found: {mets_path}")

    download_all(
        mets_path=mets_path,
        output_dir=args.output,
        viewer_url=args.viewer,
    )


if __name__ == "__main__":
    main()