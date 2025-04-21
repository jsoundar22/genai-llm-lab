import sqlite3
import os

# Remove the existing database file if it exists to avoid schema conflicts
if os.path.exists("retail_store.db"):
    os.remove("retail_store.db")

# Connect to the sql db, create it if it doesn't exist
conn = sqlite3.connect("retail_store.db")
cursor = conn.cursor()
# Create the products table
# Create the products table
cursor.execute("""
CREATE TABLE IF NOT EXISTS products(
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL
)
""")
# Create the staff table
cursor.execute("""
CREATE TABLE IF NOT EXISTS staff(
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL
)
""")
# Create the orders table
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders(
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    staff_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
)
""")
#Insert some data into the products table
cursor.executemany("""
                    INSERT INTO products (product_name, price) VALUES (?, ?)
                    """, 
                    [
                        ("Laptop", 1000.00),
                        ("Smartphone", 500.00),
                        ("Tablet", 300.00),
                        ("Monitor", 200.00),
                        ("Headphones", 50.00)
                    ])
#Insert into staff table
cursor.executemany("""
                    INSERT INTO staff (first_name, last_name) VALUES (?, ?)
                    """, 
                    [
                        ("John", "Doe"),
                        ("Jane", "Smith"),
                        ("Emily", "Jones"),
                        ("Michael", "Brown"),
                        ("Sarah", "Davis")
                    ])
#Insert into orders table
cursor.executemany("""
                    INSERT INTO orders (customer_name, staff_id, product_id) VALUES (?, ?, ?)
                    """, 
                    [
                        ("Alice", 1, 1),
                        ("Bob", 2, 2),
                        ("Charlie", 3, 3),
                        ("David", 4, 4),
                        ("Eve", 5, 5)
                    ])
# Commit the changes and close the connection
conn.commit()
conn.close()
# Check if the database file was created
if os.path.exists("retail_store.db"):
    print("Database created successfully.")
else:
    print("Failed to create the database.")