# AutoVoice Frontend

Modern React + TypeScript frontend for AutoVoice singing voice conversion system.

## 🚀 Quick Start

### Prerequisites
- Node.js >= 18.0.0
- npm >= 9.0.0

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Layout.tsx       # Main layout wrapper
│   │   ├── VoiceProfileSelector.tsx
│   │   └── SingingConversion/
│   │       ├── UploadInterface.tsx
│   │       ├── ConversionControls.tsx
│   │       └── ProgressDisplay.tsx
│   ├── pages/               # Page components
│   │   ├── HomePage.tsx
│   │   ├── SingingConversionPage.tsx
│   │   ├── VoiceProfilesPage.tsx
│   │   └── SystemStatusPage.tsx
│   ├── services/            # API and WebSocket services
│   │   ├── api.ts           # REST API client
│   │   └── websocket.ts     # WebSocket client
│   ├── App.tsx              # Root component
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── vite.config.ts           # Vite config
└── tailwind.config.js       # TailwindCSS config
```

## 🛠️ Tech Stack

- **React 18.2** - UI framework
- **TypeScript** - Type safety
- **Vite 5.0** - Build tool
- **TailwindCSS 3.3** - Styling
- **React Router 6** - Routing
- **React Query** - Data fetching
- **Socket.IO Client** - WebSocket
- **Axios** - HTTP client
- **Wavesurfer.js** - Audio waveforms
- **Chart.js** - Pitch graphs
- **Lucide React** - Icons

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
VITE_API_URL=/api/v1
VITE_WS_URL=http://localhost:5000
```

### Backend Proxy

The Vite dev server proxies API requests to the Flask backend:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:5000',
    '/socket.io': {
      target: 'http://localhost:5000',
      ws: true,
    },
  },
}
```

## 📦 Available Scripts

- `npm run dev` - Start development server (port 3000)
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

## 🎨 Features

### Singing Voice Conversion
- Drag-and-drop file upload
- Voice profile selection
- Real-time conversion progress
- Pitch shift control (-12 to +12 semitones)
- Preservation settings (pitch, vibrato, expression)
- Quality presets (fast, balanced, high, studio)
- Audio playback and download

### Voice Profile Management
- Create voice profiles from samples
- Edit profile metadata
- Delete profiles
- Preview voice samples

### System Monitoring
- GPU status and utilization
- Memory usage tracking
- Model loading status
- Real-time updates

## 🔌 API Integration

### REST API

```typescript
import { apiService } from './services/api'

// Convert song
const result = await apiService.convertSong(
  audioFile,
  profileId,
  settings
)

// Get voice profiles
const profiles = await apiService.getVoiceProfiles()
```

### WebSocket

```typescript
import { wsService } from './services/websocket'

// Connect
await wsService.connect()

// Subscribe to job updates
wsService.subscribeToJob(jobId, {
  onProgress: (progress) => console.log(progress),
  onComplete: (result) => console.log(result),
  onError: (error) => console.error(error),
})
```

## 🚀 Deployment

### Production Build

```bash
npm run build
```

Output: `dist/` directory

### Serve with Flask

The Flask backend can serve the built frontend:

```python
# src/auto_voice/web/app.py
app = Flask(__name__, static_folder='../../frontend/dist')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
```

### Docker

```dockerfile
FROM node:18 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12
# ... copy frontend/dist to Flask static folder
```

## 📝 Development Notes

- Hot module replacement (HMR) enabled
- TypeScript strict mode enabled
- ESLint + Prettier configured
- TailwindCSS JIT mode
- Code splitting for optimal loading

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port in vite.config.ts
server: { port: 3001 }
```

### Backend Connection Issues
```bash
# Check backend is running
curl http://localhost:5000/api/v1/health

# Update VITE_WS_URL in .env
VITE_WS_URL=http://localhost:5000
```

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📄 License

Part of the AutoVoice project.

