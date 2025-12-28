# VidIntel Pro - Advanced Video Intelligence Platform

VidIntel Pro is a comprehensive AI-powered video analysis platform that provides deep insights into video content through advanced machine learning and computer vision techniques. The platform offers real-time video processing, content moderation, engagement analytics, and automated subtitle generation.

## 🚀 Features

### Core Analysis Capabilities
- **Speech-to-Text Transcription** - High-accuracy transcription using OpenAI Whisper
- **Content Summarization** - AI-generated summaries with multiple formats (bullet points, executive, technical)
- **Sentiment Analysis** - Real-time emotional analysis with timeline tracking
- **Named Entity Recognition** - Automatic identification of people, places, organizations
- **Topic Extraction** - Key topic identification and categorization
- **Language Detection** - Multi-language support with confidence scoring

### Advanced Video Processing
- **Scene Change Detection** - Automatic identification of scene transitions
- **Face Detection & Recognition** - Computer vision-based face analysis
- **Text Recognition (OCR)** - Extract text from video frames
- **Speaker Diarization** - Identify and separate different speakers
- **Audio Feature Analysis** - Comprehensive audio characteristic extraction

### Content Intelligence
- **Content Moderation** - Toxicity detection and safety scoring
- **Engagement Metrics** - Predict viewer engagement potential
- **Accessibility Scoring** - Evaluate content accessibility compliance
- **Chapter Generation** - Automatic video chapter creation
- **Highlight Extraction** - Identify key moments based on sentiment

### Platform Features
- **Batch Processing** - Process multiple videos simultaneously
- **Subtitle Generation** - Automatic SRT file creation with timestamps
- **Content Similarity** - Find similar videos using semantic analysis
- **Analytics Dashboard** - Comprehensive insights and reporting
- **Translation Support** - Multi-language text translation
- **Rate Limiting** - Built-in API protection and usage controls

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Framework**: FastAPI with async support
- **AI/ML Stack**: 
  - OpenAI Whisper (speech recognition)
  - Transformers (BERT, BART models)
  - Sentence Transformers (embeddings)
  - KeyBERT (keyword extraction)
  - SpaCy (NLP processing)
- **Computer Vision**: OpenCV, face_recognition, pytesseract
- **Audio Processing**: librosa, pydub, moviepy
- **Database**: SQLite with comprehensive schema
- **Deployment**: Docker containerization

### Frontend (React)
- **Framework**: React 19 with modern hooks
- **Routing**: React Router DOM
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI, Lucide React icons
- **Charts**: Recharts for analytics visualization
- **HTTP Client**: Axios for API communication

## 📋 Prerequisites

### Backend Requirements
- Python 3.10+
- FFmpeg (for video/audio processing)
- System dependencies for computer vision
- At least 4GB RAM (8GB+ recommended for optimal performance)

### Frontend Requirements
- Node.js 16+
- npm or yarn package manager

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd vidintel-pro
```

2. **Set up Python environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
python -m spacy download en_core_web_sm
```

5. **Initialize database**
```bash
python -c "from main import init_db; init_db()"
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

## 🚀 Running the Application

### Development Mode

1. **Start the backend server**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the frontend development server**
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Production Deployment

#### Using Docker

1. **Build and run backend**
```bash
cd backend
docker build -t vidintel-backend .
docker run -p 8000:8000 vidintel-backend
```

2. **Build frontend**
```bash
cd frontend
npm run build
```

## 📊 API Endpoints

### Core Analysis
- `POST /analyze/comprehensive` - Full video analysis with all features
- `POST /analyze/quick` - Fast analysis for basic insights
- `POST /batch/analyze` - Process multiple videos

### Content Operations
- `POST /translate` - Text translation
- `POST /generate/summary` - Custom summary generation
- `POST /generate/chapters` - Video chapter creation
- `POST /extract/highlights` - Key moment extraction

### Content Moderation
- `POST /analyze/content_moderation` - Safety and toxicity analysis
- `POST /analyze/engagement_metrics` - Engagement prediction

### Data & Analytics
- `GET /analytics/dashboard/{video_id}` - Comprehensive analytics
- `GET /stats/platform` - Platform usage statistics
- `POST /search/similar` - Find similar content

### File Operations
- `GET /download/subtitles/{video_id}.srt` - Download subtitle files
- `GET /download/thumbnail/{video_id}.jpg` - Download thumbnails

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model paths
WHISPER_MODEL_PATH=/path/to/models
TRANSFORMERS_CACHE=/path/to/cache

# Database configuration
DATABASE_URL=sqlite:///vidintel.db

# API configuration
MAX_FILE_SIZE=100MB
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=3600
```

### Model Configuration
The platform automatically downloads required AI models on first run. Models include:
- Whisper base model (~140MB)
- BART summarization model (~1.6GB)
- Sentiment analysis models (~500MB)
- NER models (~1.3GB)

## 📈 Usage Examples

### Basic Video Analysis
```python
import requests

# Upload and analyze video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze/quick',
        files={'file': f}
    )
    
analysis = response.json()
print(f"Summary: {analysis['summary']}")
print(f"Sentiment: {analysis['sentiment']}")
```

### Batch Processing
```python
files = [
    ('files', open('video1.mp4', 'rb')),
    ('files', open('video2.mp4', 'rb'))
]

response = requests.post(
    'http://localhost:8000/batch/analyze',
    files=files,
    data={'analysis_type': 'quick'}
)
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📁 Project Structure

```
vidintel-pro/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── train.py             # Training and model management
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Container configuration
│   ├── static_subtitles/   # Generated subtitle files
│   └── vidintel.db         # SQLite database
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API services
│   │   └── utils/          # Utility functions
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
└── README.md
```

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse with configurable limits
- **Content Moderation**: Built-in toxicity and safety detection
- **Input Validation**: Comprehensive request validation
- **File Type Restrictions**: Secure file upload handling
- **CORS Configuration**: Proper cross-origin resource sharing

## 🎯 Performance Optimization

- **Async Processing**: Non-blocking video analysis
- **Model Caching**: Efficient AI model management
- **Batch Operations**: Optimized multi-file processing
- **Background Tasks**: Cleanup and maintenance operations
- **Database Indexing**: Optimized query performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs` endpoint
- Review the comprehensive logging output for debugging

## 🔮 Roadmap

- [ ] Real-time video streaming analysis
- [ ] Advanced face recognition with identity tracking
- [ ] Multi-language subtitle generation
- [ ] Cloud storage integration (AWS S3, Google Cloud)
- [ ] Advanced analytics with ML insights
- [ ] Mobile app development
- [ ] Enterprise SSO integration
- [ ] Custom model training interface

---

**VidIntel Pro** - Transforming video content into actionable intelligence through the power of AI.