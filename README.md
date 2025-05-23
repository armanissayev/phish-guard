# Phishing URL Detector

A web application that detects whether a given URL is a phishing site or not using machine learning.

## Project Structure

The project consists of two main components:

### Backend (Server)
- Python-based server with a machine learning model
- RESTful API endpoints for URL analysis
- ML model trained on phishing URL dataset

### Frontend (Client)
- React-based web interface
- Modern, responsive design
- Real-time URL analysis

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

## Development

- Backend runs on: http://localhost:5000
- Frontend runs on: http://localhost:3000

## Technologies Used

- Backend:
  - Python
  - Flask/FastAPI
  - Scikit-learn/TensorFlow
  - Pandas
  - NumPy

- Frontend:
  - React
  - TypeScript
  - Material-UI/Tailwind CSS
  - Axios 