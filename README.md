# 🎵 Emotion Music App

A web application that analyzes emotional trajectories in Thai songs using Natural Language Processing and Machine Learning.

## 🌟 Features

- **Emotion Analysis**: Analyzes emotional content in song lyrics using Thai NLP
- **Interactive Visualization**: Creates interactive emotion trajectory charts using Plotly
- **YouTube Integration**: Fetches song metadata from YouTube API
- **Smart Search**: Search songs by emotional patterns (e.g., "เศร้า → หวัง")
- **Thai Language Support**: Full support for Thai language processing
- **Vector Search**: Semantic search using FAISS and sentence transformers

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
├── app.py                 # Main Flask application
├── emotion_model.py       # Emotion detection using transformers
├── nlp_utils.py          # Thai text preprocessing
├── youtube_utils.py      # YouTube API integration
├── vectorstore.py        # FAISS vector search
├── analysis.py           # Emotion trajectory visualization
├── search.py             # Search functionality
├── db_setup.py           # Database initialization
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── songs.db              # SQLite database
├── templates/            # HTML templates
│   ├── layout.html
│   ├── index.html
│   ├── search.html
│   ├── song_detail.html
│   ├── explore.html
│   └── dashboard.html
└── static/               # Static assets
    └── *.png            # Song thumbnails
```

## 🎯 How It Works

### 1. Song Analysis Pipeline
- **Input**: YouTube URL + Lyrics
- **Processing**: 
  - Extract metadata from YouTube API
  - Segment lyrics into meaningful parts
  - Analyze emotion for each segment using BART model
  - Create interactive emotion trajectory visualization
- **Output**: Stored in SQLite database with visualization

### 2. Emotion Detection
- Uses `facebook/bart-large-mnli` for zero-shot classification
- Supports 8 emotion categories: sad, lonely, hope, happy, excited, calm, angry, neutral
- Thai lexicon fallback for better accuracy
- Threshold-based classification (default: 0.55)

### 3. Search System
- **Emotion Pattern Search**: Find songs by emotional progression (e.g., "เศร้า → หวัง")
- **Semantic Search**: Vector-based search using multilingual sentence transformers
- **Soft Subsequence Matching**: Flexible pattern matching for emotional sequences

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Add new songs and view all songs |
| `/search` | GET/POST | Search songs by emotion patterns |
| `/song/<id>` | GET | View detailed song information |
| `/song/<id>/refresh` | GET | Refresh song analysis |
| `/song/<id>/rebuild` | POST | Rebuild song analysis |
| `/song/<id>/delete` | POST | Delete song |
| `/explore` | GET | Explore popular emotions and transitions |
| `/dashboard` | GET | View application statistics |

## 🎨 Features in Detail

### Emotion Trajectory Visualization
- Interactive Plotly charts showing emotional progression
- Hover effects with detailed information
- Responsive design for all devices

### Thai Language Support
- Uses PyThaiNLP for Thai text tokenization
- Custom emotion aliases mapping (e.g., "เศร้า", "เสียใจ" → "sad")
- Handles Thai text preprocessing and cleaning

### Smart Search
- **Pattern Search**: "เศร้า → หวัง" finds songs that transition from sadness to hope
- **Flexible Matching**: Supports various connectors (→, ->, ถึง, ไป, แล้ว)
- **Canonical Mapping**: Normalizes different emotion expressions

## 🛠️ Configuration

### Environment Variables
```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### Model Configuration
- **Emotion Model**: `facebook/bart-large-mnli`
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Search**: FAISS IndexFlatL2 with 384 dimensions

## 📊 Database Schema

### Songs Table
- `id`, `title`, `youtube_link`, `description`, `tags`
- `upload_date`, `view_count`, `like_count`, `lyrics`
- `graph_html` (stored Plotly visualization)

### Segments Table
- `id`, `song_id`, `segment_order`, `text`, `emotion`
- Links to songs table with foreign key

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
