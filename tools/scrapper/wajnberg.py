import sys
from typing import List
import requests
import json
from tqdm import tqdm
import os

MAX_PAGE_NUMBER_TO_RECORD = 10


class Main:
    def __init__(self):
        self.path_url = "https://wajnberg.com"
        self.music_style = "wajnberg"
        self.urls: List[str] = []
        self.mode = sys.argv[1]
        if self.mode == "import":
            self.fetch_data()
        elif self.mode == "download":
            self.download_data()
        else:
            print("Invalid mode. Use 'import' or 'download'.")
            sys.exit(1)

    def fetch_data(self):
        response = requests.get(self.path_url)
        self.get_mp3_url(response.text)
        self.save_url_json(f"{self.music_style}.json")

    def get_mp3_url(self, website_content: json) -> None:
        split_content = website_content.split(".mp3")
        for i in range(len(split_content) - 1):
            content = split_content[i].split("https://wajnberg.com")[-1]
            url = "https://wajnberg.com" + content + ".mp3"
            if url not in self.urls:
                self.urls.append(url)

    def download_data(self):
        if not os.path.exists("music"):
            os.makedirs("music")
        self.open_url_json(f"{self.music_style}.json")
        if not os.path.exists(f"music/{self.music_style}/"):
            os.makedirs(f"music/{self.music_style}/")
        for url in tqdm(self.urls):
            file_name = url.split("/")[-1]
            response = requests.get(url)
            with open(f"music/{self.music_style}/{file_name}", "wb") as file:
                file.write(response.content)

    def save_url_json(self, output_file: str):
        with open(output_file, "w") as file:
            json.dump(self.urls, file)

    def open_url_json(self, output_file: str):
        with open(output_file, "r") as file:
            self.urls = json.load(file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrapper.py <mode>")
        sys.exit(1)
    main = Main()
