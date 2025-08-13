# Sentiview - Real-time Sentiment Analysis Platform

A comprehensive web application for real-time sentiment analysis of text and Amazon product reviews, built with Next.js and FastAPI.

## 🚀 Features

- **Text Sentiment Analysis**: Analyze sentiment of any text input with confidence scores
- **Amazon Reviews Analysis**: Real-time sentiment analysis of Amazon product reviews
- **Interactive Visualizations**: Pie charts, timelines, and word clouds
- **Machine Learning Pipeline**: Custom ML models with TextBlob fallback
- **Responsive Design**: Modern UI built with Next.js and Tailwind CSS
- **RESTful API**: FastAPI backend with comprehensive endpoints

## 🏗️ Architecture

\`\`\`
sentiview/
├── app/                    # Next.js frontend
├── components/             # React components
├── lib/                   # Utilities and API client
├── services/
│   └── api/               # FastAPI backend
└── docs/                  # Documentation
\`\`\`

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Recharts** - Data visualization
- **React Query** - API state management

### Backend
- **FastAPI** - Python web framework
- **scikit-learn** - Machine learning
- **NLTK & spaCy** - Natural language processing
- **TextBlob** - Sentiment analysis fallback
- **Pandas** - Data processing for Amazon reviews
- **Uvicorn** - ASGI server

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker (optional)

### Local Development

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd sentiview
   \`\`\`

2. **Setup Frontend**
   \`\`\`bash
   npm install
   npm run dev
   \`\`\`

3. **Setup Backend**
   \`\`\`bash
   cd services/api
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   \`\`\`

4. **Environment Variables**
   \`\`\`bash
   # Frontend (.env.local)
   NEXT_PUBLIC_API_URL=http://localhost:8000
   
   # Backend (services/api/.env)
   # No external API keys required - uses publicly available datasets
   \`\`\`

### Docker Deployment

\`\`\`bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
\`\`\`

## 📖 API Documentation

The API documentation is available at `http://localhost:8000/docs` when running the backend.

### Key Endpoints

- `POST /analyze` - Analyze text sentiment
- `POST /analyze/batch` - Batch text analysis
- `GET /reviews/analyze` - Analyze Amazon reviews data
- `GET /health` - Health check
- `GET /model/info` - Model information

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes |

### Model Training

Train a custom sentiment model using Amazon reviews data:

\`\`\`bash
cd services/api
python scripts/train.py
\`\`\`

## 🚀 Deployment

### Vercel (Frontend)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push

### Railway/Render (Backend)
1. Connect your repository
2. Set Python buildpack
3. Configure environment variables
4. Deploy the `services/api` directory

## 🧪 Testing

\`\`\`bash
# Frontend tests
npm test

# Backend tests
cd services/api
pytest
\`\`\`

## 📊 Performance

- **Frontend**: Optimized with Next.js App Router and React Query
- **Backend**: Async FastAPI with efficient ML pipeline
- **Caching**: Model caching and API response optimization
- **Monitoring**: Health checks and error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support, please open an issue on GitHub or contact the development team.
