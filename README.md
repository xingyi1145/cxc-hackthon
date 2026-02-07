# README.md

# Housing Affordability Stress Tester

This project is a Flask web application designed to help users assess their housing affordability based on their financial data. It utilizes Auth0 for user authentication and SQLite with SQLAlchemy for persistent data storage.

## Project Structure

```
cxc-hackthon
├── static
│   └── css
│       └── style.css
├── templates
│   ├── dashboard.html
│   ├── index.html
│   └── layout.html
├── .env.template
├── app.py
├── requirements.txt
└── README.md
```

## Features

- User authentication using Auth0
- SQLite database for storing user financial data
- Dashboard for users to view and update their financial information
- Responsive design with CSS styling

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd cxc-hackthon
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Copy the `.env.template` to `.env` and fill in the required values:

   ```bash
   cp .env.template .env
   ```

   Update the `.env` file with your Auth0 credentials and database URI.

5. **Run the application:**

   ```bash
   python app.py
   ```

6. **Access the application:**

   Open your web browser and go to `http://localhost:5000`.

## Usage

- Navigate to the landing page to log in using Auth0.
- After logging in, you will be redirected to the dashboard where you can view and update your financial data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.