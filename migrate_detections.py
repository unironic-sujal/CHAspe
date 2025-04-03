from app import app, db, DetectionResult, User
from sqlalchemy import create_engine, text

def migrate_detections():
    # Create engine for the old database
    old_engine = create_engine('sqlite:///instance/users.db')
    
    # Get all detection results from the old database
    with old_engine.connect() as conn:
        old_detections = conn.execute(text('SELECT * FROM detection_result')).fetchall()
        
        # Migrate each detection to the new database
        for detection in old_detections:
            # Get username from user table
            user = User.query.get(detection.user_id)
            username = user.username if user else "Unknown User"
            
            # Create new detection record
            new_detection = DetectionResult(
                id=detection.id,
                user_id=detection.user_id,
                username=username,
                original_image=detection.original_image,
                processed_image=detection.processed_image,
                crater_count=detection.crater_count,
                roughness_index=detection.roughness_index,
                created_at=detection.created_at,
                settings=detection.settings
            )
            db.session.add(new_detection)
        
        # Commit all changes
        db.session.commit()
        print(f"Successfully migrated {len(old_detections)} detections")

if __name__ == '__main__':
    with app.app_context():
        migrate_detections() 