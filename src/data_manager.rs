use std::fs;
use std::fs::{File, OpenOptions};
use std::io::Write;

pub fn load_text_to_file(file_path: &str) -> String {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    contents
}

pub fn import_text_to_file(file_path: &str, content: String) -> std::io::Result<()> {
    let mut file = File::create(file_path)?;
    file.write_all(content.as_bytes())
}

pub fn add_text_to_file(file_path: String, content: String) -> std::io::Result<()> {
    let mut file = OpenOptions::new().append(true).open(file_path)?;
    file.write_all(content.as_bytes())
}
