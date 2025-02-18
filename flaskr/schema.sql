-- Create the database (if not already created)
PRAGMA foreign_keys = ON;

-- Drop existing table if it exists
DROP TABLE IF EXISTS query_responses;

-- Create the table to store the questions and responses
CREATE TABLE query_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    response TEXT NOT NULL
);