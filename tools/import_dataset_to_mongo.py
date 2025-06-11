import os
import sys
import pymongo
from tqdm import tqdm
import base64


class Main:
    def __init__(self):
        self.dataset_folder_path = sys.argv[1]
        self.db = self.create_database_if_not_exists()

    @staticmethod
    def create_database_if_not_exists():
        client = pymongo.MongoClient("mongodb://mongo:pass@localhost:27017/")
        db = client["dataset_db"]
        if "dataset_collection" not in db.list_collection_names():
            db.create_collection("dataset_collection")
        return db

    def compute(self):
        dataset = []
        folders = os.listdir(self.dataset_folder_path)
        for folder in tqdm(folders):
            folder_path = os.path.join(self.dataset_folder_path, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    image_path = os.path.join(folder_path, file)
                    with open(image_path, "rb") as image_file:
                        imag_str = base64.b64encode(image_file.read())
                    dataset.append(
                        {"category": folder, "image_data": imag_str.decode("utf-8")}
                    )
        self.db.dataset_collection.insert_many(dataset)


if __name__ == "__main__":
    main = Main()
    main.compute()
