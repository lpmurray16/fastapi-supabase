import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")  # Should be aws-0-us-east-2.pooler.supabase.com for transaction pooler
PORT = os.getenv("DB_PORT")  # Should be 6543 for transaction pooler
DBNAME = os.getenv("DB_NAME")

# Connect to the database
def connect_to_db():
    try:
        print(f"Attempting to connect to PostgreSQL database at {HOST}:{PORT}...")
        print(f"Database: {DBNAME}, User: {USER}")
        
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Please check your .env file and ensure your database credentials are correct.")
        print("Also verify that the database server is running and accessible from your network.")
        return None

# Example usage
if __name__ == "__main__":
    connection = connect_to_db()
    if connection:
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()
        
        # Example query
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        print("Current Time:", result)

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("Connection closed.")