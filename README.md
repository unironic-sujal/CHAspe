# CHAspe

A Flask web application for crater detection and analysis with user authentication, article management, and more.

## Features

- User authentication and management
- Crater detection algorithm using OpenCV
- Article/blog management system
- Contact form with admin response capabilities
- E-commerce functionality with order management

## Technology Stack

- Flask web framework
- SQLAlchemy ORM
- OpenCV for image processing
- Flask-Login for authentication
- Flask-Migrate for database migrations

## Installation

1. Clone the repository
```
git clone https://github.com/yourusername/CHAspe.git
cd CHAspe
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Set up environment variables
Create a `.env` file with the following variables:
```
EMAIL_ADDRESS=your-email@example.com
EMAIL_PASSWORD=your-email-password
```

4. Initialize the database
```
python init_db.py
```

5. Run the application
```
flask run
```

## Deployment

The application can be deployed on platforms like PythonAnywhere. See deployment guide in the documentation.

## Project Structure

- `app.py`: Main application file
- `models.py`: Database models
- `init_db.py`: Database initialization script
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, images)
- `migrations/`: Database migration files 