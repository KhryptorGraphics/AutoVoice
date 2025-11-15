import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { SingingConversionPage } from './pages/SingingConversionPage'
import { VoiceProfilesPage } from './pages/VoiceProfilesPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { BatchConversionPage } from './pages/BatchConversionPage'
import { ConversionHistoryPage } from './pages/ConversionHistoryPage'
import { SettingsPage } from './pages/SettingsPage'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/singing-conversion" element={<SingingConversionPage />} />
            <Route path="/batch-conversion" element={<BatchConversionPage />} />
            <Route path="/history" element={<ConversionHistoryPage />} />
            <Route path="/voice-profiles" element={<VoiceProfilesPage />} />
            <Route path="/system-status" element={<SystemStatusPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  )
}

export default App

