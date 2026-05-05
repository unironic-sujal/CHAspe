from app import app, db, User
from werkzeug.security import generate_password_hash
import sys

def set_admin(email):
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"No user found with email: {email}")
            sys.exit(1)
        user.is_admin = True
        db.session.commit()
        print(f"✅ '{user.username}' ({email}) is now an admin.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python set_admin.py <user-email>")
        sys.exit(1)
    set_admin(sys.argv[1])
