import requests
import os

def download_pdf(url, dest_path):
    resp = requests.get(url)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {dest_path}")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    reports = {
        "2023": "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NYSE_ALL_2023.pdf",
        "2022": "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NYSE_ALL_2022.pdf",
    }

    for year, url in reports.items():
        try:
            download_pdf(url, f"data/raw/Allstate_{year}_10K.pdf")
        except Exception as e:
            print(f"Failed to download {year} report: {e}")
