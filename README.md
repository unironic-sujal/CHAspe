# CHAspe 🌕

A powerful Flask web application for lunar surface analysis, featuring automated crater detection using OpenCV, integrated e-commerce for merchandise, a publishing platform for space articles, and an advanced administrative dashboard.

## 🚀 Key Features

* **Crater Detection AI:** Upload lunar surface images to automatically detect, measure, and analyze craters using OpenCV and advanced image processing techniques.
* **Google OAuth Authentication:** Secure, seamless user login and registration powered by Google OpenID Connect.
* **E-Commerce Shop:** A built-in merchandise shop with a cart system, order tracking, and dynamic order status updates.
* **Publishing Platform:** Administrators can write, edit, and publish articles/blogs directly to the site.
* **Secure Admin Dashboard:** A dedicated backend for administrators to manage users, process orders, review detection logs, and reply to contact messages.
* **Native Gmail API Integration:** Reliable, anti-spam-compliant email delivery for newsletters and auto-replies directly from the platform's Gmail account using OAuth2 tokens.

## 🛠️ Technology Stack

* **Backend:** Python, Flask, SQLAlchemy (ORM)
* **Database:** PostgreSQL
* **Image Processing:** OpenCV (`opencv-python-headless`), NumPy
* **Authentication:** Authlib (Google OAuth), Flask-Login
* **Security:** Flask-WTF (CSRF Protection), Flask-Limiter (Rate Limiting), Werkzeug Security
* **Email Delivery:** Google API Client (`gmail.send` scope)
* **Deployment:** Render (Web Service)

## 🔒 Security & Architecture

This application is hardened for production use:
- **Global CSRF Protection:** All mutating routes and forms are protected against Cross-Site Request Forgery.
- **Rate Limiting:** Brute-force protection on authentication, contact, and subscription routes.
- **Environment Isolation:** All secrets, API keys, and database URIs are isolated in environment variables.
- **Role-Based Access Control:** Strict `@login_required` and `is_admin` boolean checks on all sensitive endpoints.
- **Proxy Middleware:** Configured with `ProxyFix` to correctly handle HTTPS resolution securely behind Render's load balancers.

## 💻 Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/unironic-sujal/CHAspe.git
   cd CHAspe
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   # Flask Security
   SECRET_KEY=your-long-random-secret-key

   # Database (PostgreSQL string)
   DATABASE_URL=postgresql://user:password@localhost/chaspe

   # Initial Admin Credentials
   ADMIN_EMAIL=your-admin-email@example.com
   ADMIN_PASSWORD=your-secure-password

   # Google OAuth (For User Sign-In)
   GOOGLE_CLIENT_ID=your-oauth-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-oauth-client-secret

   # Gmail API (For Outbound Emails)
   EMAIL_ADDRESS=chaspe.newsletter@gmail.com
   GMAIL_TOKEN_JSON={"token": "..."}
   ```

4. **Initialize the Database**
   ```bash
   python init_db.py
   ```

5. **Run the application**
   ```bash
   flask run
   ```

## ☁️ Production Deployment (Render)

1. Connect the repository to a Render Web Service.
2. Set the **Build Command** to: `./build.sh`
3. Set the **Start Command** to: `gunicorn app:app`
4. Add all environment variables listed above to the Render Dashboard.
5. **Important:** CHAspe requires a persistent PostgreSQL database instance, as local SQLite databases will be wiped upon Render container restarts.