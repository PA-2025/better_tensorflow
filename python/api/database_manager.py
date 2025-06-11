import sqlite3
import pymongo


class DatabaseManager:
    @staticmethod
    def get_training_data():
        con = sqlite3.connect("app_database.db")
        cur = con.cursor()
        res = cur.execute("select distinct training_name from training_data ")
        rows = res.fetchall()
        names = [row[0] for row in rows]
        final = []

        for file_name in names:
            results = []
            res = cur.execute(
                "select mse, accuracy from training_data  where training_name = ? order by epoch",
                (file_name,),
            )
            for line in res.fetchall():
                results.append(
                    {
                        "mse": line[0],
                        "accuracy": line[1],
                    }
                )

            final.append(
                {"name": f"{file_name}_mse", "data": [x["mse"] for x in results]}
            )
            final.append(
                {
                    "name": f"{file_name}_accuracy",
                    "data": [x["accuracy"] for x in results],
                }
            )
        return final

    @staticmethod
    def get_results():
        con = sqlite3.connect("app_database.db")
        cur = con.cursor()
        res = cur.execute("select distinct training_name from training_data ")
        rows = res.fetchall()
        results = [row[0] for row in rows]
        files = []
        for r in results:
            files.append(f"{r}_mse")
            files.append(f"{r}_accuracy")
        return files

    @staticmethod
    def get_training_data_mongo():
        try:
            client = pymongo.MongoClient(
                "mongodb://mongo:pass@localhost:27017/",
                connectTimeoutMS=500,
                serverSelectionTimeoutMS=500,
                socketTimeoutMS=500,
            )
            db = client["db"]
            collection = db["training_data"]
            res = collection.find({}, {"_id": 0, "training_name": 1})
            names = [r["training_name"] for r in res]
            final = []
            for file_name in names:
                results = []
                res = collection.find(
                    {"training_name": file_name},
                    {"_id": 0, "epoch": 1, "mse": 1, "accuracy": 1},
                ).sort("epoch")
                for line in res:
                    results.append(
                        {
                            "mse": line["mse"],
                            "accuracy": line["accuracy"],
                        }
                    )

                final.append(
                    {"name": f"{file_name}_mse", "data": [x["mse"] for x in results]}
                )
                final.append(
                    {
                        "name": f"{file_name}_accuracy",
                        "data": [x["accuracy"] for x in results],
                    }
                )
            return final

        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return []

    @staticmethod
    def get_results_mongo():
        try:
            client = pymongo.MongoClient(
                "mongodb://mongo:pass@localhost:27017/",
                connectTimeoutMS=500,
                serverSelectionTimeoutMS=500,
                socketTimeoutMS=500,
            )
            db = client["db"]
            collection = db["training_data"]
            res = collection.find({}, {"_id": 0, "training_name": 1})
            files = []
            for r in res:
                files.append(f"{r['training_name']}_mse")
                files.append(f"{r['training_name']}_accuracy")
            return files
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return []
