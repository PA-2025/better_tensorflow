import sys
from typing import List
import requests
import json
from tqdm import tqdm
import os

MAX_PAGE_NUMBER_TO_RECORD = 40


class Main:
    def __init__(self):
        self.path_url = "https://www.bensound.com/royalty-free-music"
        self.music_style = ["rock", "pop", "jazz", "classical", "hip-hop"]
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
        for style in self.music_style:
            for i in tqdm(range(1, MAX_PAGE_NUMBER_TO_RECORD)):
                url = f"{self.path_url}?page={i}&tag[]={style}"
                response = requests.get(url)
                self.get_mp3_url(response.text)
            self.save_url_json(f"{style}.json")
            self.urls = []

    def get_mp3_url(self, website_content: json) -> None:
        split_content = website_content.split("cdn.bensound.com")
        for i in range(1, len(split_content)):
            extension = split_content[i].split(".")[1].split('"')[0]
            if extension == "mp3":
                url = "https://cdn.bensound.com" + split_content[i].split('"')[0][1:]
                if url not in self.urls:
                    self.urls.append(url)

    def download_data(self):
        if not os.path.exists("music"):
            os.makedirs("music")
        for style in self.music_style:
            self.open_url_json(f"{style}.json")
            if not os.path.exists(f"music/{style}/"):
                os.makedirs(f"music/{style}/")
            for url in tqdm(self.urls):
                file_name = url.split("/")[-1]
                response = requests.get(url)
                with open(f"music/{style}/{file_name}", "wb") as file:
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
