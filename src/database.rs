use mongodb::{
    bson::{doc, Document},
    Client, Collection,
};
use rusqlite::Connection;

fn create_database() -> rusqlite::Result<()> {
    let conn = Connection::open("app_database.db")?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_data(id integer primary key autoincrement, training_name text, mse real, accuracy real, epoch int);",
        [],
    )?;

    Ok(())
}

pub fn insert_training_score(
    training_name: String,
    mse: f32,
    accuracy: f32,
    epoch: i32,
) -> rusqlite::Result<()> {
    if std::env::var("USE_MONGO").unwrap_or_default() == "1" {
        insert_training_score_mongo(training_name, mse, accuracy, epoch);
        return Ok(());
    }

    create_database().expect("Error during database creation");

    let conn = Connection::open("app_database.db")?;

    conn.execute(
        "insert into training_data (training_name, mse, accuracy, epoch) values (?1, ?2, ?3, ?4)",
        [
            training_name,
            mse.to_string(),
            accuracy.to_string(),
            epoch.to_string(),
        ],
    )?;

    Ok(())
}
#[tokio::main]
async fn insert_training_score_mongo(
    training_name: String,
    mse: f32,
    accuracy: f32,
    epoch: i32,
) -> mongodb::error::Result<()> {
    println!("Inserting training score into MongoDB");
    let client = Client::with_uri_str("mongodb://mongo:pass@localhost:27017/").await?;
    let db = client.database("db");
    if !db
        .list_collection_names()
        .await?
        .contains(&"training_data".to_string())
    {
        db.create_collection("training_data").await?;
    }
    let collection: Collection<Document> = db.collection("training_data");
    let doc = doc! {
        "training_name": training_name,
        "mse": mse,
        "accuracy": accuracy,
        "epoch": epoch,
    };
    collection
        .insert_one(doc)
        .await
        .map_err(|e| mongodb::error::Error::from(e))?;
    Ok(())
}
