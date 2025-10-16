# 🎵 Emotion Music App

A sophisticated web application that analyzes the emotional journey of songs using Natural Language Processing and Machine Learning. The app processes song lyrics to create interactive emotion trajectory visualizations, helping users understand how emotions flow throughout a song.

## Key Capabilities
- **Bilingual Processing**: Full support for Thai, English, and mixed-language lyrics
- **Smart Analysis**: Advanced emotion detection using BART model with lexicon fallback
- **Interactive Visualization**: Dynamic Plotly charts showing emotional progression
- **Color-coded Lyrics**: Lyrics segments displayed with emotion-specific colors and icons
- **Overall Emotion Summary**: Comprehensive emotion analysis with detailed explanations
- **Intelligent Search**: Natural language queries and emotion pattern matching
- **YouTube Integration**: Automatic metadata extraction and view/like tracking
- **Real-time Processing**: Instant emotion analysis and visualization generation

## 🌟 Features

### 🎯 Core Functionality
- **Song Analysis Pipeline**: Add YouTube URL + lyrics → Automatic emotion analysis → Interactive visualization
- **Emotion Detection**: 8 emotion categories (sad, lonely, hope, happy, excited, calm, angry, neutral)
- **Smart Segmentation**: Automatic detection of song sections (intro, verse, chorus, bridge, outro)
- **Color-coded Display**: Each emotion has unique colors and icons for easy identification
- **Overall Emotion Analysis**: Comprehensive emotion summary with natural language explanations
- **Real-time Visualization**: Interactive Plotly charts with hover effects and responsive design

### 🔍 Advanced Search Capabilities
- **Emotion Pattern Search**: Find songs by emotional progression (e.g., "เศร้า → หวัง" or "sad → hope")
- **Natural Language Queries**: Thai and English natural language search support
- **Flexible Matching**: Arrow format (→), soft subsequence matching, and constant emotion detection
- **Bilingual Support**: Automatic Thai-English emotion conversion and canonical mapping

### 🌐 Language Processing
- **Mixed Language Support**: Simultaneous Thai-English text processing
- **Auto-tokenization**: Smart word boundary detection for mixed-language lyrics
- **Lexicon Fallback**: Comprehensive Thai emotion lexicon with English mapping
- **PyThaiNLP Integration**: Advanced Thai language tokenization

### 📊 Data Management
- **SQLite Database**: Efficient storage with songs and segments tables
- **YouTube API Integration**: Automatic metadata, view count, and like tracking
- **Graph Caching**: Stored interactive visualizations for fast loading
- **CRUD Operations**: Full song management with refresh and rebuild capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- YouTube API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emotion-music-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "YOUTUBE_API_KEY=your_youtube_api_key_here" > .env
   ```

4. **Initialize the database**
   ```bash
   python db_setup.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
emotion-music-app/
├── app.py                 # Main Flask application with routing and business logic
├── emotion_model.py       # BART-based emotion detection with Thai-English mapping
├── nlp_utils.py          # Advanced text preprocessing and auto-tokenization
├── youtube_utils.py      # YouTube API integration for metadata extraction
├── vectorstore.py        # FAISS vector search for semantic similarity
├── analysis.py           # Interactive Plotly visualization generation
├── search.py             # Advanced search with emotion pattern matching
├── db_setup.py           # SQLite database schema initialization
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (YouTube API key)
├── songs.db              # SQLite database (auto-generated)
├── templates/            # Jinja2 HTML templates
│   ├── layout.html       # Base template with navigation
│   ├── index.html        # Main page for adding and viewing songs
│   ├── search.html       # Advanced search interface
│   ├── song_detail.html  # Detailed song view with segments
│   ├── explore.html      # Popular emotions and transitions
│   └── dashboard.html    # Application statistics and metrics
└── README.md             # Project documentation
```

## 🎯 How It Works

### 1. Song Analysis Pipeline
- **Input**: YouTube URL + Lyrics (Thai/English)
- **Processing**: 
  - Extract metadata from YouTube API
  - Segment lyrics into meaningful parts (intro, verse, chorus, etc.)
  - Analyze emotion for each segment using BART model
  - Create interactive emotion trajectory visualization with English labels
- **Output**: Stored in SQLite database with bilingual visualization

### 2. Emotion Detection
- **Primary Model**: `facebook/bart-large-mnli` for zero-shot classification
- **8 Emotion Categories** with bidirectional Thai-English mapping:
  - sad (เศร้า) - includes เสียใจ, หม่น, หมอง, หดหู่, ซึม, ร้องไห้, ทุกข์, น้อยใจ, ผิดหวัง
  - lonely (เหงา) - includes เดียวดาย, ว้าเหว่
  - hope (หวัง) - includes ความหวัง, มีหวัง, กำลังใจ, สู้, พยายาม
  - happy (สุข) - includes ยินดี, ดีใจ, ร่าเริง, สดใส, สนุก, ยิ้ม, เบิกบาน
  - excited (ตื่นเต้น) - includes เร้าใจ, พีค, มัน, เปรี้ยว, ฮึกเหิม
  - calm (สงบ) - includes เยือกเย็น, นิ่ง, ใจเย็น, ผ่อนคลาย, ชิล
  - angry (โกรธ) - includes โมโห, เดือด, แค้น, เคือง
  - neutral (เฉย) - includes ปกติ, ธรรมดา
- **Intelligent Fallback**: Comprehensive Thai lexicon with automatic English conversion
- **Configurable Threshold**: Default 0.55 confidence score for classification
- **Multi-label Support**: Optional multi-label emotion detection

### 3. Advanced Search System
- **Emotion Pattern Search**: 
  - Arrow format: "เศร้า → หวัง" or "sad → hope"
  - Natural language: "เพลงที่เริ่มเศร้าแล้วค่อยๆเปลี่ยนเป็นหวัง"
  - Constant emotion: "เพลงที่อารมณ์ neutral ตลอดทั้งเพลง"
- **Intelligent Parsing**: 
  - Complex query analysis with transition word detection
  - Canonical emotion mapping and alias resolution
  - Intensity and transition pattern recognition
- **Flexible Matching Algorithms**:
  - Soft subsequence matching for emotional progressions
  - Constant emotion detection for stable songs
  - Bilingual query normalization
- **Semantic Search**: FAISS-powered vector search using multilingual sentence transformers

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Main page: Add new songs and view all existing songs |
| `/search` | GET/POST | Advanced search with emotion pattern matching |
| `/song/<id>` | GET | Detailed song view with segments and interactive visualization |
| `/song/<id>/refresh` | GET | Re-analyze song with current emotion model |
| `/song/<id>/rebuild` | POST | Complete rebuild of song analysis and visualization |
| `/song/<id>/delete` | POST | Delete song and all associated data |
| `/explore` | GET | Discover popular emotions, transitions, and stable songs |
| `/dashboard` | GET | Application metrics and emotion statistics |
| `/tokenize` | POST | API endpoint for automatic text tokenization |

## 🎨 Features in Detail

### Emotion Trajectory Visualization
- Interactive Plotly charts showing emotional progression
- English labels for better international understanding
- Axis labels: "Step" and "Emotion"
- Hover effects with detailed information
- Responsive design for all devices

### Color-coded Emotion System
- **SAD**: Blue background with blue icon 💙
- **LONELY**: Purple background with purple icon 💜
- **HOPE**: Green background with green icon 💚
- **HAPPY**: Yellow background with yellow icon 💛
- **EXCITED**: Red background with red icon ❤️
- **CALM**: Indigo background with blue icon 🔵
- **ANGRY**: Orange background with orange icon 🧡
- **NEUTRAL**: Gray background with white icon ⚪

### Overall Emotion Analysis
- Calculates dominant emotion from all song segments
- Explains why the song has that overall emotion with natural language descriptions
- Shows secondary emotions when present
- Detailed explanations only visible in song detail view

### Bilingual Processing Engine
- **Advanced Tokenization**:
  - PyThaiNLP for Thai word boundary detection
  - NLTK for English text processing
  - Auto-tokenization API endpoint for mixed-language text
  - Real-time tokenization in web interface
- **Smart Section Detection**:
  - Thai patterns: อินโทร, ท่อน, คอรัส, บริดจ์, เอาท์โทร
  - English patterns: intro, verse, chorus, bridge, outro
  - Fallback to paragraph and length-based segmentation
- **Emotion Mapping System**:
  - Canonical emotion labels in English for database consistency
  - Comprehensive Thai alias dictionary with 50+ emotion words
  - Bidirectional conversion with automatic canonicalization
  - Context-aware emotion detection from complex phrases

### Intelligent Search Features
- **Multi-format Pattern Support**:
  - Arrow format: "เศร้า → หวัง", "sad → hope", "เศร้า -> หวัง"
  - Natural language: "เพลงที่เริ่มเศร้าแล้วค่อยๆเปลี่ยนเป็นหวัง"
  - Single emotion: "neutral", "เศร้า"
  - Constant emotion: "เพลงที่อารมณ์ neutral ตลอดทั้งเพลง"
- **Advanced Query Processing**:
  - Transition word detection (เริ่ม, ค่อยๆ, พุ่ง, เปลี่ยน, กลาย)
  - Intensity recognition (มาก, เบาๆ, ค่อยๆ, พุ่ง)
  - Complex emotion phrase parsing
  - Automatic single-emotion to progression inference
- **Smart Matching Algorithms**:
  - Soft subsequence matching for flexible pattern detection
  - Constant emotion detection for stable emotional songs
  - Normalized comparison with canonical emotion labels

## 🛠️ Configuration

### Environment Variables
```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### Technical Configuration
- **Primary Emotion Model**: `facebook/bart-large-mnli` (Zero-shot classification)
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Search**: FAISS IndexFlatL2 with 384-dimensional vectors
- **NLP Libraries**: PyThaiNLP 4.1.0, NLTK 3.8.1, Transformers 4.35.2
- **Visualization**: Plotly 5.17.0 for interactive charts
- **Database**: SQLite with songs and segments tables
- **Web Framework**: Flask 2.3.3 with Jinja2 templates

## 📊 Database Schema

### Songs Table
```sql
CREATE TABLE songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    youtube_link TEXT,
    description TEXT,
    tags TEXT,
    upload_date TEXT,
    view_count INTEGER,
    like_count INTEGER,
    lyrics TEXT,
    image_path TEXT,
    graph_html TEXT  -- Cached Plotly visualization
);
```

### Segments Table
```sql
CREATE TABLE segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    segment_order INTEGER,
    text TEXT,
    emotion TEXT,  -- English emotion labels (sad, happy, etc.)
    FOREIGN KEY(song_id) REFERENCES songs(id)
);
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Transformers](https://huggingface.co/transformers/) for emotion detection models
- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) for Thai language processing
- [Plotly](https://plotly.com/) for interactive visualizations
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Flask](https://flask.palletsprojects.com/) for web framework
