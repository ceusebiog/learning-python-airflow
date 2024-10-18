CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    date_of_birth DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    account_type VARCHAR(50) NOT NULL,
    balance DECIMAL(15, 2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INT REFERENCES accounts(account_id),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(15, 2) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    description TEXT
);

INSERT INTO customers (name, email, date_of_birth) VALUES
('John Doe', 'john.doe@example.com', '1985-05-15'),
('Jane Smith', 'jane.smith@example.com', '1990-10-20');

INSERT INTO accounts (customer_id, account_type, balance) VALUES
(1, 'savings', 5000),
(2, 'checking', 3000);

INSERT INTO transactions (account_id, amount, transaction_type, status, description) VALUES
(1, 1500, 'deposit', 'completed', 'Salary deposit'),
(1, -200, 'withdrawal', 'completed', 'ATM withdrawal'),
(2, 500, 'deposit', 'completed', 'Freelance payment'),
(2, -1000, 'withdrawal', 'completed', 'Online shopping');