use rusqlite::Connection;


fn create_database() -> rusqlite::Result<()> {
    let conn = Connection::open("app_database.db")?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_data(id integer primary key autoincrement, training_name text, mse real, accuracy real, epoch int);",
        [],
    )?;

    Ok(())
}

pub fn insert_training_score(training_name: String, mse: f32, accuracy: f32, epoch: i32) -> rusqlite::Result<()> {
    create_database().expect("Error during database creation");

    let conn = Connection::open("app_database.db")?;

    conn.execute("insert into training_data (training_name, mse, accuracy, epoch) values (?1, ?2, ?3, ?4)",
                 [training_name, mse.to_string(), accuracy.to_string(), epoch.to_string()])?;

    Ok(())
}
