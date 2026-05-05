import os
from app import app, db, User
from werkzeug.security import generate_password_hash

def init_database():
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")

        # Auto-promote admin user if ADMIN_EMAIL is set
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')

        if admin_email:
            user = User.query.filter_by(email=admin_email).first()
            if user:
                # User already exists — promote to admin
                if not user.is_admin:
                    user.is_admin = True
                    db.session.commit()
                    print(f"✅ Promoted existing user '{user.username}' to admin.")
                else:
                    print(f"ℹ️  '{user.username}' is already an admin.")
            elif admin_password:
                # User doesn't exist — create admin account
                new_admin = User(
                    username='admin',
                    email=admin_email,
                    password=generate_password_hash(admin_password),
                    is_admin=True
                )
                db.session.add(new_admin)
                db.session.commit()
                print(f"✅ Created new admin account for {admin_email}.")
            else:
                print(f"⚠️  ADMIN_EMAIL set but no user found and no ADMIN_PASSWORD to create one.")
        else:
            print("ℹ️  No ADMIN_EMAIL set — skipping admin setup.")

if __name__ == '__main__':
    init_database()