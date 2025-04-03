from app import app, db
from sqlalchemy import create_engine, text

def cleanup_users_db():
    # Create engine for the database
    engine = create_engine('sqlite:///instance/users.db')
    
    # Drop the detection_result table if it exists
    with engine.connect() as conn:
        conn.execute(text('DROP TABLE IF EXISTS detection_result'))
        conn.commit()
        print("Successfully cleaned up users database")

if __name__ == '__main__':
    with app.app_context():
        cleanup_users_db() 