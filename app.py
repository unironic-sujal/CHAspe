from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, current_user, logout_user, LoginManager
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import shutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from flask_mail import Mail, Message
import json
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_BINDS'] = {
    'detections': 'sqlite:///detections.db',
    'contacts': 'sqlite:///contacts.db',  # New database for contact messages
    'articles': 'sqlite:///articles.db',  # New database for articles
    'orders': 'sqlite:///orders.db'  # New database for orders
}

# Email Configuration
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', 'chaspe.newsletter@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Updated upload folder structure
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ORIGINAL_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'original')
app.config['PROCESSED_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['ORIGINAL_FOLDER'], 
               app.config['PROCESSED_FOLDER'],
               os.path.join(app.config['UPLOAD_FOLDER'], 'articles')]:
    os.makedirs(folder, exist_ok=True)

# Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = EMAIL_ADDRESS
app.config['MAIL_PASSWORD'] = EMAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = EMAIL_ADDRESS

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = 'auth'
mail = Mail(app)

# Database model for Users
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    
    @property
    def is_admin(self):
        return self.username == 'admin' and self.email == 'admin'

# Database model for Detection Results
class DetectionResult(db.Model):
    __bind_key__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(100), nullable=False)  # Store username for reference
    original_image = db.Column(db.String(255), nullable=False)
    processed_image = db.Column(db.String(255), nullable=False)
    crater_count = db.Column(db.Integer, nullable=False)
    roughness_index = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    settings = db.Column(db.JSON)

# Database model for Contact Messages
class ContactMessage(db.Model):
    __bind_key__ = 'contacts'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(20), default='pending')
    replied_at = db.Column(db.DateTime, nullable=True)

# Database model for Articles
class Article(db.Model):
    __bind_key__ = 'articles'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    subtitle = db.Column(db.String(200))
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500))  # URL for article cover image
    author = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20), default='draft')  # draft, published, archived
    category = db.Column(db.String(50))  # e.g., news, tutorial, research
    tags = db.Column(db.String(200))  # Comma-separated tags
    views = db.Column(db.Integer, default=0)  # Track number of views

class Order(db.Model):
    __bind_key__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    order_number = db.Column(db.String(20), unique=True, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    shipping_fee = db.Column(db.Float, nullable=False)
    tax_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, processing, shipped, delivered, cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=False)
    
    # Contact Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    
    # Shipping Address
    address = db.Column(db.String(200), nullable=False)
    apartment = db.Column(db.String(100))
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    zipcode = db.Column(db.String(20), nullable=False)
    country = db.Column(db.String(100), nullable=False)

class OrderItem(db.Model):
    __bind_key__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    size = db.Column(db.String(20))

class OrderStatusHistory(db.Model):
    __bind_key__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_craters(image_path):
    # Read the image
    img = cv2.imread(image_path)
    original = img.copy()
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Create different scales for crater detection
    scales = [(0.5, 15), (1.0, 25), (2.0, 35)]
    all_craters = []
    
    for scale, blur_size in scales:
        # Resize image for this scale
        if scale != 1.0:
            current = cv2.resize(denoised, None, fx=scale, fy=scale)
        else:
            current = denoised.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(current, (blur_size, blur_size), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 10, 50)
        
        # Dilate edges to connect broken contours
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Scale back contour points if needed
            if scale != 1.0:
                contour = (contour / scale).astype(np.int32)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size
            area = cv2.contourArea(contour)
            min_area = 100  # Minimum crater area
            max_area = width * height * 0.1  # Maximum 10% of image
            
            if area < min_area or area > max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # More lenient circularity threshold for larger craters
            min_circularity = 0.4 if area > 5000 else 0.5
            if circularity < min_circularity:
                continue
            
            # Check intensity gradient
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), -1)
            
            # Calculate mean intensity inside and outside the contour
            inside_mean = cv2.mean(gray, mask=mask)[0]
            
            # Dilate mask for outside region
            outside_mask = cv2.dilate(mask, kernel, iterations=3) - mask
            outside_mean = cv2.mean(gray, mask=outside_mask)[0]
            
            # Check if there's significant intensity difference
            if abs(inside_mean - outside_mean) < 10:
                continue
            
            # Store valid crater
            all_craters.append({
                'contour': contour,
                'bounds': (x, y, w, h),
                'area': area,
                'confidence': circularity * abs(inside_mean - outside_mean) / 255
            })
    
    # Remove overlapping detections
    filtered_craters = []
    all_craters.sort(key=lambda x: x['area'], reverse=True)
    
    for crater in all_craters:
        x1, y1, w1, h1 = crater['bounds']
        overlapping = False
        
        for existing in filtered_craters:
            x2, y2, w2, h2 = existing['bounds']
            
            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            min_area = min(w1 * h1, w2 * h2)
            
            if overlap_area > 0.3 * min_area:
                overlapping = True
                break
        
        if not overlapping:
            filtered_craters.append(crater)
    
    # Draw results
    crater_count = len(filtered_craters)
    total_crater_area = 0
    
    for crater in filtered_craters:
        x, y, w, h = crater['bounds']
        area = crater['area']
        total_crater_area += area
        
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw contour
        cv2.drawContours(img, [crater['contour']], -1, (0, 0, 255), 2)
        
        # Add size measurement
        cv2.putText(img, f'w={w}px', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Calculate roughness index
    image_area = height * width
    roughness_index = (total_crater_area / image_area) * 100
    
    # Add summary information
    cv2.putText(img, f'Craters: {crater_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Roughness: {roughness_index:.2f}%', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save processed image
    processed_path = image_path.replace('.', '_processed.')
    cv2.imwrite(processed_path, img)
    
    return processed_path, crater_count, roughness_index
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        if 'signup' in request.form:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already exists! Please log in instead.', 'danger')
                return redirect(url_for('auth'))
            
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('auth'))
        
        if 'signin' in request.form:
            email = request.form['email']
            password = request.form['password']
            
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('home'))
            else:
                flash('Invalid login credentials! Please try again.', 'danger')
                return redirect(url_for('auth'))
    
    return render_template('auth.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=current_user.username)

@app.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    if request.method == 'POST':
        if 'surface_image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['surface_image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Get user-specific upload paths
                user_original, user_processed = get_user_upload_path(current_user.id)
                print(f"User upload paths - Original: {user_original}, Processed: {user_processed}")
                
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = secure_filename(file.filename)
                unique_filename = f"{timestamp}_{filename}"
                print(f"Generated filename: {unique_filename}")
                
                # Save original image
                original_path = os.path.join(user_original, unique_filename)
                file.save(original_path)
                print(f"Saved original image to: {original_path}")
                
                # Process the image
                processed_path, crater_count, roughness_index = detect_craters(original_path)
                print(f"Processed image saved to: {processed_path}")
                
                # Move processed image to user's processed folder
                processed_filename = os.path.basename(processed_path)
                new_processed_path = os.path.join(user_processed, processed_filename)
                shutil.move(processed_path, new_processed_path)
                print(f"Moved processed image to: {new_processed_path}")
                
                # Save detection result with correct paths
                original_db_path = os.path.join('original', str(current_user.id), unique_filename).replace('\\', '/')
                processed_db_path = os.path.join('processed', str(current_user.id), processed_filename).replace('\\', '/')
                print(f"Database paths - Original: {original_db_path}, Processed: {processed_db_path}")
                
                detection = DetectionResult(
                    user_id=current_user.id,
                    username=current_user.username,
                    original_image=original_db_path,
                    processed_image=processed_db_path,
                    crater_count=crater_count,
                    roughness_index=roughness_index,
                    settings={
                        'min_area': 100,
                        'max_area': 0.1,
                        'min_circularity': 0.4,
                        'scales': [(0.5, 15), (1.0, 25), (2.0, 35)]
                    }
                )
                db.session.add(detection)
                db.session.commit()
                print("Detection result saved to database")
                
                # Cleanup old images
                cleanup_old_images(current_user.id)
                
                return render_template('scan.html', 
                                     original_image=detection.original_image,
                                     processed_image=detection.processed_image,
                                     crater_count=crater_count,
                                     roughness_index=roughness_index,
                                     detection_id=detection.id)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                flash('Error processing image. Please try again.', 'error')
                return redirect(url_for('scan'))
    
    return render_template('scan.html')

@app.route('/detections')
@login_required
def detections():
    user_detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.created_at.desc()).all()
    return render_template('detections.html', detections=user_detections, timedelta=timedelta)

@app.route('/export/<int:detection_id>')
@login_required
def export_detection(detection_id):
    detection = DetectionResult.query.get_or_404(detection_id)
    if detection.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('detections'))
    
    import json
    from flask import Response
    
    data = {
        'id': detection.id,
        'created_at': detection.created_at.isoformat(),
        'crater_count': detection.crater_count,
        'roughness_index': detection.roughness_index,
        'settings': detection.settings
    }
    
    return Response(
        json.dumps(data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment;filename=detection_{detection_id}.json'}
    )

@app.route('/profile')
@login_required
def profile():
    user_detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.created_at.desc()).all()
    
    # Get user's orders
    user_orders = Order.query.filter_by(user_id=current_user.id).order_by(Order.created_at.desc()).all()
    
    return render_template('profile.html', 
                         username=current_user.username, 
                         email=current_user.email,
                         detections=user_detections,
                         orders=user_orders,
                         timedelta=timedelta)

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/merch_shop')
@login_required
def merch_shop():
    return render_template('merch_shop.html')

@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    if request.method == 'POST':
        try:
            # Generate a unique order number
            order_number = f"CHO-{int(time.time())}-{current_user.id}"
            
            # Get form data
            first_name = request.form.get('firstName')
            last_name = request.form.get('lastName')
            email = request.form.get('email')
            phone = request.form.get('phone')
            address = request.form.get('address')
            apartment = request.form.get('apartment', '')
            city = request.form.get('city')
            state = request.form.get('state')
            zipcode = request.form.get('zipCode')
            country = request.form.get('country')
            payment_method = request.form.get('paymentMethod')
            
            # Get cart items from the form (sent as JSON string)
            cart_items = json.loads(request.form.get('cartItems'))
            
            # Calculate totals
            subtotal = sum(float(item['price'].replace('$', '')) * int(item['quantity']) for item in cart_items)
            shipping_fee = 5.99
            tax_amount = subtotal * 0.07
            total_amount = subtotal + shipping_fee + tax_amount
            
            # Create order in database
            new_order = Order(
                user_id=current_user.id,
                order_number=order_number,
                total_amount=total_amount,
                shipping_fee=shipping_fee,
                tax_amount=tax_amount,
                status='pending',
                payment_method=payment_method,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                address=address,
                apartment=apartment,
                city=city,
                state=state,
                zipcode=zipcode,
                country=country
            )
            
            db.session.add(new_order)
            db.session.commit()
            
            # Record initial status in history
            status_history = OrderStatusHistory(
                order_id=new_order.id,
                status='pending',
                notes='Order placed'
            )
            db.session.add(status_history)
            
            # Add order items
            for item in cart_items:
                order_item = OrderItem(
                    order_id=new_order.id,
                    product_name=item['title'],
                    price=float(item['price'].replace('$', '')),
                    quantity=int(item['quantity']),
                    size=item['size']
                )
                db.session.add(order_item)
            
            db.session.commit()
            
            # Return success response
            flash('Order placed successfully!', 'success')
            
            # Send order confirmation email
            try:
                order_confirmation_email = Message(
                    'Your CHAspe Order Confirmation',
                    recipients=[email],
                    body=f"""Dear {first_name} {last_name},

Thank you for your order! We're processing it now and will ship your items soon.

Order Number: {order_number}
Date: {datetime.utcnow().strftime('%B %d, %Y')}

Order Summary:
{'='*50}
"""
                )
                
                # Add order items to the email
                item_details = ""
                for item in cart_items:
                    price = float(item['price'].replace('$', ''))
                    item_total = price * int(item['quantity'])
                    item_details += f"\n{item['title']} - Size: {item['size']} × {item['quantity']} = ${item_total:.2f}"
                
                order_confirmation_email.body += f"{item_details}\n{'='*50}\n"
                order_confirmation_email.body += f"""
Subtotal: ${subtotal:.2f}
Shipping: ${shipping_fee:.2f}
Tax: ${tax_amount:.2f}
Total: ${total_amount:.2f}

Shipping to:
{address}{', ' + apartment if apartment else ''}
{city}, {state} {zipcode}
{country}

You can track your order status at any time by logging into your account.

Thank you for shopping with CHAspe!

Best regards,
The CHAspe Team
"""
                
                mail.send(order_confirmation_email)
                print(f"Order confirmation email sent to {email}")
            except Exception as email_error:
                print(f"Error sending order confirmation email: {str(email_error)}")
                # Don't show error to user or stop the order process
            
            return redirect(url_for('order_confirmation', order_id=new_order.id))
            
        except Exception as e:
            db.session.rollback()
            print(f"Error placing order: {str(e)}")
            flash('There was an error processing your order. Please try again.', 'error')
            return redirect(url_for('checkout'))
    
    return render_template('checkout.html')

@app.route('/order-confirmation/<int:order_id>')
@login_required
def order_confirmation(order_id):
    order = Order.query.get_or_404(order_id)
    
    # Only allow order owners and admins to view order details
    if order.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this order.', 'error')
        return redirect(url_for('home'))
    
    items = OrderItem.query.filter_by(order_id=order.id).all()
    
    # Get the order status history
    status_history = OrderStatusHistory.query.filter_by(order_id=order.id).order_by(OrderStatusHistory.timestamp).all()
    
    return render_template('order_confirmation.html', order=order, items=items, status_history=status_history)

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth'))

@app.route('/subscribe', methods=['POST'])
def subscribe():
    subscriber_email = request.form.get('email')
    if not subscriber_email:
        flash('Please provide an email address.', 'error')
        return redirect(url_for('home'))
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = subscriber_email
        msg['Subject'] = 'Welcome to CHAspe Newsletter!'

        body = '''Thank you for subscribing to the CHAspe newsletter!

We're excited to keep you updated with the latest developments in lunar surface analysis and our platform's features.

Best regards,
The CHAspe Team'''

        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            text = msg.as_string()
            server.send_message(msg)

        flash('Thank you for subscribing! Please check your email.', 'success')
            
    except Exception as e:
        flash('There was an error processing your subscription. Please try again later.', 'error')
        print(f"Email error: {str(e)}")
    
    return redirect(url_for('home'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject')
            message = request.form.get('message')

            if not all([name, email, subject, message]):
                flash('Please fill in all fields.', 'error')
                return redirect(url_for('contact'))
                
            # Validate subject from dropdown
            allowed_subjects = ['Bug Report', 'Suggestion', 'Feedback', 'Question', 'Feature Request', 'Other']
            if subject not in allowed_subjects:
                flash('Please select a valid subject.', 'error')
                return redirect(url_for('contact'))

            # Create new contact message
            new_message = ContactMessage(
                name=name,
                email=email,
                subject=subject,
                message=message,
                user_id=current_user.id if current_user.is_authenticated else None,
                status='pending'
            )
            
            # Save to database first
            try:
                db.session.add(new_message)
                db.session.commit()
                print("Message saved to database successfully")
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
                db.session.rollback()
                flash('Error saving message to database. Please try again.', 'error')
                return redirect(url_for('contact'))
            
            # Try to send auto-reply email
            try:
                if EMAIL_PASSWORD:  # Only try to send email if password is configured
                    auto_reply = Message(
                        'Thank you for contacting CHAspe',
                        recipients=[email],
                        body=f"""Dear {name},

Thank you for reaching out to us. We have received your message and will get back to you within 24-48 hours.

Your message details:
Subject: {subject}

Best regards,
CHAspe Team"""
                    )
                    mail.send(auto_reply)
                    print("Auto-reply email sent successfully")
            except Exception as email_error:
                print(f"Email error: {str(email_error)}")
                # Don't show error to user since message was saved
                pass

            flash('Your message has been sent successfully! We will get back to you soon.', 'success')
        except Exception as e:
            print(f"Contact submission error: {str(e)}")  # Log the error
            db.session.rollback()
            flash('There was an error sending your message. Please try again later.', 'error')
        
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

def get_user_upload_path(user_id):
    """Create and return user-specific upload paths"""
    user_original = os.path.join(app.config['ORIGINAL_FOLDER'], str(user_id))
    user_processed = os.path.join(app.config['PROCESSED_FOLDER'], str(user_id))
    os.makedirs(user_original, exist_ok=True)
    os.makedirs(user_processed, exist_ok=True)
    return user_original, user_processed

def cleanup_old_images(user_id, keep_count=10):
    """Keep only the most recent images for a user"""
    user_original, user_processed = get_user_upload_path(user_id)
    
    # Get all files in both directories
    original_files = sorted([f for f in os.listdir(user_original) if os.path.isfile(os.path.join(user_original, f))],
                          key=lambda x: os.path.getctime(os.path.join(user_original, x)),
                          reverse=True)
    processed_files = sorted([f for f in os.listdir(user_processed) if os.path.isfile(os.path.join(user_processed, f))],
                           key=lambda x: os.path.getctime(os.path.join(user_processed, x)),
                           reverse=True)
    
    # Remove old files
    for files, folder in [(original_files, user_original), (processed_files, user_processed)]:
        for old_file in files[keep_count:]:
            try:
                os.remove(os.path.join(folder, old_file))
            except Exception as e:
                print(f"Error removing old file {old_file}: {e}")

# Function to check if user is admin
def is_admin():
    return current_user.is_authenticated and current_user.email == 'chaspe.newsletter@gmail.com'

@app.route('/articles')
def articles():
    # Get published articles only
    articles_list = Article.query.filter_by(status='published').order_by(Article.created_at.desc()).all()
    return render_template('articles.html', articles=articles_list)

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    article = Article.query.get_or_404(article_id)
    # Only show published articles to regular users
    if article.status != 'published' and not is_admin():
        flash('Article not found', 'error')
        return redirect(url_for('articles'))
    
    # Increment view count
    article.views += 1
    db.session.commit()
    
    return render_template('article_detail.html', article=article, is_admin=is_admin())

@app.route('/admin/articles', methods=['GET', 'POST'])
@login_required
def admin_articles():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        subtitle = request.form.get('subtitle')
        content = request.form.get('content')
        category = request.form.get('category')
        tags = request.form.get('tags')
        status = request.form.get('status')
        
        # Handle image upload
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'articles', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                image_url = url_for('static', filename=f'uploads/articles/{filename}')
        
        article = Article(
            title=title,
            subtitle=subtitle,
            content=content,
            category=category,
            tags=tags,
            status=status,
            image_url=image_url,
            author=current_user.username
        )
        
        db.session.add(article)
        db.session.commit()
        
        flash('Article created successfully!', 'success')
        return redirect(url_for('admin_articles'))
    
    articles = Article.query.order_by(Article.created_at.desc()).all()
    return render_template('admin/articles.html', articles=articles)

@app.route('/admin/articles/<int:article_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_article(article_id):
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('home'))
    
    article = Article.query.get_or_404(article_id)
    
    if request.method == 'POST':
        article.title = request.form.get('title')
        article.subtitle = request.form.get('subtitle')
        article.content = request.form.get('content')
        article.category = request.form.get('category')
        article.tags = request.form.get('tags')
        article.status = request.form.get('status')
        
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                # Delete old image if exists
                if article.image_url:
                    old_image_path = os.path.join(app.root_path, 'static', article.image_url.lstrip('/'))
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'articles', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                article.image_url = url_for('static', filename=f'uploads/articles/{filename}')
        
        db.session.commit()
        flash('Article updated successfully!', 'success')
        return redirect(url_for('admin_articles'))
    
    return render_template('admin/edit_article.html', article=article)

@app.route('/admin/articles/<int:article_id>/delete', methods=['POST'])
@login_required
def delete_article(article_id):
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('home'))
    
    article = Article.query.get_or_404(article_id)
    
    # Delete associated image if exists
    if article.image_url:
        image_path = os.path.join(app.root_path, 'static', article.image_url.lstrip('/'))
        if os.path.exists(image_path):
            os.remove(image_path)
    
    db.session.delete(article)
    db.session.commit()
    
    flash('Article deleted successfully!', 'success')
    return redirect(url_for('admin_articles'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if it's the admin credentials
        if email == 'admin' and password == 'admin':
            # Create or get admin user
            admin_user = User.query.filter_by(email='admin').first()
            if not admin_user:
                admin_user = User(
                    username='admin',
                    email='admin',
                    password=generate_password_hash('admin')
                )
                db.session.add(admin_user)
                db.session.commit()
            
            login_user(admin_user)
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin')
def admin_dashboard():
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    # Get statistics
    total_users = User.query.count()
    total_scans = DetectionResult.query.count()
    
    # Count orders
    total_orders = Order.query.count()
    
    # Get contact messages without using count()
    contact_messages = []
    try:
        contact_messages = ContactMessage.query.all()
        total_messages = len(contact_messages)
    except Exception as e:
        total_messages = 0
        print(f"Error fetching messages: {str(e)}")
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_scans=total_scans,
                         total_messages=total_messages,
                         total_orders=total_orders,
                         contact_messages=contact_messages)

@app.route('/admin/reply/<int:message_id>', methods=['POST'])
def admin_reply_message(message_id):
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('auth'))

    message = ContactMessage.query.get_or_404(message_id)
    reply_text = request.form.get('reply')

    if not reply_text:
        flash('Reply cannot be empty.', 'error')
        return redirect(url_for('admin_dashboard'))

    try:
        # Send email reply
        msg = Message(
            subject=f"Re: {message.subject}",
            recipients=[message.email],
            body=reply_text
        )
        mail.send(msg)

        # Update message status
        message.status = 'replied'
        message.replied_at = datetime.utcnow()
        db.session.commit()

        flash('Reply sent successfully!', 'success')
    except Exception as e:
        flash(f'Error sending reply: {str(e)}', 'error')

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/logout')
def admin_logout():
    logout_user()
    flash('Admin logged out successfully.', 'success')
    return redirect(url_for('admin_login'))

def find_landing_spots(image_path):
    """
    Analyze the image to find potential landing spots based on crater density and surface roughness.
    Returns a list of coordinates for potential landing spots.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate local variance (roughness) using a sliding window
    window_size = 50
    stride = 25
    landing_spots = []
    
    for y in range(0, img.shape[0] - window_size, stride):
        for x in range(0, img.shape[1] - window_size, stride):
            window = blurred[y:y+window_size, x:x+window_size]
            variance = np.var(window)
            
            # Calculate local crater density (if we have crater data)
            # For now, we'll use variance as a proxy for crater density
            # Lower variance means flatter surface
            
            # Criteria for good landing spot:
            # 1. Low variance (flat surface)
            # 2. Not too close to edges
            # 3. Sufficient distance from other landing spots
            if variance < 100 and x > window_size and y > window_size:
                # Check distance from other landing spots
                is_far_enough = True
                for spot in landing_spots:
                    dist = np.sqrt((x - spot[0])**2 + (y - spot[1])**2)
                    if dist < window_size * 2:  # Minimum distance between spots
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    landing_spots.append((x + window_size//2, y + window_size//2))
    
    return landing_spots

@app.route('/landing_spots')
@login_required
def landing_spots():
    # Get user's detections
    detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.created_at.desc()).all()
    return render_template('landing_spots.html', detections=detections, timedelta=timedelta)

@app.route('/analyze_landing_spots/<int:detection_id>')
@login_required
def analyze_landing_spots(detection_id):
    # Get the detection
    detection = DetectionResult.query.get_or_404(detection_id)
    
    # Verify ownership
    if detection.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('landing_spots'))
    
    try:
        # Get the processed image path - using correct path structure
        processed_image_path = os.path.join('static', 'uploads', detection.processed_image)
        print(f"Processing image at path: {processed_image_path}")
        
        # Analyze landing spots
        landing_spots = find_landing_spots(processed_image_path)
        
        # Read the original image
        img = cv2.imread(processed_image_path)
        if img is None:
            raise ValueError(f"Could not read image at {processed_image_path}")
        
        # Draw landing spots on the image
        for spot in landing_spots:
            x, y = spot
            # Draw a green circle for each landing spot
            cv2.circle(img, (x, y), 30, (0, 255, 0), 2)
            # Add a label
            cv2.putText(img, f"Landing Spot", (x-40, y-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Get user-specific upload paths
        user_original, user_processed = get_user_upload_path(current_user.id)
        
        # Save the marked image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        marked_filename = f"landing_spots_{timestamp}.jpg"
        marked_path = os.path.join(user_processed, marked_filename)
        
        print(f"Saving marked image to: {marked_path}")
        cv2.imwrite(marked_path, img)
        
        # Create correct path for database
        processed_db_path = os.path.join('processed', str(current_user.id), marked_filename).replace('\\', '/')
        
        # Update the detection record with the new processed image that includes landing spots
        try:
            detection.processed_image = processed_db_path
            db.session.commit()
            print(f"Updated detection {detection_id} with landing spots image")
        except Exception as e:
            print(f"Error updating detection record: {str(e)}")
            db.session.rollback()
        
        # Return JSON with the image URL and landing spots count
        return jsonify({
            'success': True,
            'image_url': url_for('static', filename='uploads/' + processed_db_path),
            'landing_spots_count': len(landing_spots)
        })
        
    except Exception as e:
        import traceback
        print(f"Error analyzing landing spots: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/messages')
@login_required
def admin_messages():
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    # Get page number from query parameters
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of messages per page
    
    # Get all messages with pagination
    messages_pagination = ContactMessage.query.order_by(ContactMessage.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template(
        'admin/messages.html',
        messages=messages_pagination.items,
        current_page=page,
        total_pages=messages_pagination.pages
    )

@app.route('/admin/messages/<int:message_id>/delete', methods=['POST'])
@login_required
def delete_contact_message(message_id):
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    message = ContactMessage.query.get_or_404(message_id)
    
    try:
        db.session.delete(message)
        db.session.commit()
        flash('Message deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting message: {str(e)}', 'error')
    
    return redirect(url_for('admin_messages'))

@app.route('/admin/orders')
@login_required
def admin_orders():
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    # Get page number from query parameters
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of orders per page
    
    # Get all orders with pagination
    orders_pagination = Order.query.order_by(Order.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template(
        'admin/orders.html',
        orders=orders_pagination.items,
        current_page=page,
        total_pages=orders_pagination.pages
    )

@app.route('/admin/orders/<int:order_id>/update', methods=['POST'])
@login_required
def update_order_status(order_id):
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    order = Order.query.get_or_404(order_id)
    new_status = request.form.get('status')
    
    # Validate status
    valid_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
    if new_status not in valid_statuses:
        flash('Invalid status value.', 'error')
        return redirect(url_for('admin_orders'))
    
    try:
        # Only record a status change if the status actually changed
        if order.status != new_status:
            # Update the order
            order.status = new_status
            
            # Record the status change in history
            status_history = OrderStatusHistory(
                order_id=order.id,
                status=new_status
            )
            db.session.add(status_history)
            db.session.commit()
            
            # Send status update email to customer
            try:
                status_update_email = Message(
                    f'Your CHAspe Order {order.order_number} Status Updated',
                    recipients=[order.email],
                    body=f"""Dear {order.first_name} {order.last_name},

The status of your order #{order.order_number} has been updated to: {new_status.upper()}

You can view the complete details of your order by logging into your account.

Thank you for shopping with CHAspe!

Best regards,
The CHAspe Team
"""
                )
                mail.send(status_update_email)
            except Exception as email_error:
                print(f"Error sending status update email: {str(email_error)}")
            
            flash(f'Order status updated to {new_status}.', 'success')
        else:
            flash('Order status unchanged.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating order: {str(e)}', 'error')
    
    return redirect(url_for('admin_orders'))

@app.route('/admin/orders/<int:order_id>/details')
@login_required
def admin_order_details(order_id):
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    order = Order.query.get_or_404(order_id)
    items = OrderItem.query.filter_by(order_id=order.id).all()
    subtotal = sum(item.price * item.quantity for item in items)
    
    # Get customer account information
    customer = User.query.get(order.user_id)
    
    # Get the order status history
    status_history = OrderStatusHistory.query.filter_by(order_id=order.id).order_by(OrderStatusHistory.timestamp).all()
    
    return render_template(
        'admin/order_details.html',
        order=order,
        items=items,
        customer=customer,
        subtotal=subtotal,
        status_history=status_history
    )

if __name__ == '__main__':
    with app.app_context():
        # Create all tables in all databases
        db.create_all()
    app.run(debug=True)