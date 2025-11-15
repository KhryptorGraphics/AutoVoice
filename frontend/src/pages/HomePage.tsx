import { Link } from 'react-router-dom'
import { Music, Zap, Shield, Cpu, ArrowRight } from 'lucide-react'

export function HomePage() {
  const features = [
    {
      icon: Music,
      title: 'Pitch Preservation',
      description: 'Maintains the exact pitch contour from the original performance with <5 cents accuracy',
    },
    {
      icon: Zap,
      title: 'Vibrato Transfer',
      description: 'Preserves vibrato rate and depth within 10% tolerance for natural singing',
    },
    {
      icon: Shield,
      title: 'Expression Intact',
      description: 'Keeps dynamics, emotional nuances, and singing talent of the original artist',
    },
    {
      icon: Cpu,
      title: 'GPU Accelerated',
      description: 'Fast processing with CUDA acceleration - convert songs in under 30 seconds',
    },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          Transform Any Song with
          <span className="text-primary-600"> AI Voice Conversion</span>
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Replace the singing voice in any song while perfectly preserving the original pitch,
          vibrato, and artistic expression. Powered by state-of-the-art So-VITS-SVC architecture.
        </p>
        <Link
          to="/singing-conversion"
          className="inline-flex items-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold px-8 py-4 rounded-lg transition-colors text-lg"
        >
          <span>Start Converting</span>
          <ArrowRight className="w-5 h-5" />
        </Link>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
        {features.map((feature) => {
          const Icon = feature.icon
          return (
            <div
              key={feature.title}
              className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow"
            >
              <div className="flex items-start space-x-4">
                <div className="p-3 bg-primary-100 rounded-lg">
                  <Icon className="w-6 h-6 text-primary-600" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600">{feature.description}</p>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* How It Works */}
      <div className="bg-white rounded-lg shadow-lg p-8 mb-16">
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            {
              step: '1',
              title: 'Upload Song',
              description: 'Upload any song file (MP3, WAV, FLAC)',
            },
            {
              step: '2',
              title: 'Select Voice',
              description: 'Choose a target voice profile',
            },
            {
              step: '3',
              title: 'Configure',
              description: 'Adjust pitch shift and quality settings',
            },
            {
              step: '4',
              title: 'Convert',
              description: 'AI processes and converts the vocals',
            },
          ].map((item) => (
            <div key={item.step} className="text-center">
              <div className="w-12 h-12 bg-primary-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                {item.step}
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{item.title}</h3>
              <p className="text-sm text-gray-600">{item.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg shadow-lg p-8 text-white">
        <h2 className="text-3xl font-bold mb-4 text-center">
          Powered by Advanced AI
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
          <div>
            <h3 className="text-2xl font-bold mb-2">So-VITS-SVC 5.0</h3>
            <p className="text-primary-100">
              State-of-the-art singing voice conversion architecture
            </p>
          </div>
          <div>
            <h3 className="text-2xl font-bold mb-2">CREPE Pitch</h3>
            <p className="text-primary-100">
              Sub-10 cent pitch accuracy for perfect preservation
            </p>
          </div>
          <div>
            <h3 className="text-2xl font-bold mb-2">HiFi-GAN</h3>
            <p className="text-primary-100">
              High-quality 44.1kHz audio synthesis
            </p>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="text-center mt-16">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Ready to Transform Your Music?
        </h2>
        <p className="text-lg text-gray-600 mb-8">
          Start converting songs with professional-quality AI voice transformation
        </p>
        <div className="flex justify-center space-x-4">
          <Link
            to="/singing-conversion"
            className="bg-primary-600 hover:bg-primary-700 text-white font-semibold px-8 py-4 rounded-lg transition-colors"
          >
            Convert a Song
          </Link>
          <Link
            to="/voice-profiles"
            className="bg-white hover:bg-gray-50 text-gray-900 font-semibold px-8 py-4 rounded-lg border-2 border-gray-300 transition-colors"
          >
            Manage Voice Profiles
          </Link>
        </div>
      </div>
    </div>
  )
}

