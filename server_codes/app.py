from flask import Flask
from routes import main_blueprint

# Initialize the Flask application
app = Flask(__name__)

# Register the blueprint
app.register_blueprint(main_blueprint)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
