import React, { useState, useEffect } from 'react';
import Header from './components/layout/Header';
import Dashboard from './components/dashboard/Dashboard';
import UploadSection from './components/upload/UploadSection';
import Analytics from './components/analytics/Analytics';
import Moderation from './components/moderation/Moderation';
import LoadingSpinner from './components/common/LoadingSpinner';
import { useAPI } from './hooks/useAPI';

const VidIntelPro = () => {
  const [activeView, setActiveView] = useState('dashboard');
  const [platformStats, setPlatformStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const api = useAPI();

  useEffect(() => {
    const loadPlatformStats = async () => {
      try {
        setLoading(true);
        const stats = await api.getPlatformStats();
        console.log('Fetched platform stats:', stats);
        setPlatformStats(stats);
      } catch (error) {
        console.error('Failed to load platform stats:', error);
      } finally {
        setLoading(false);
      }
    };

    loadPlatformStats();
  }, [api]);

  const renderActiveView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard platformStats={platformStats} />;
      case 'upload':
        return <UploadSection />;
      case 'analytics':
        return <Analytics platformStats={platformStats} />;
      case 'moderation':
        return <Moderation />;
      default:
        return <Dashboard platformStats={platformStats} />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="xl" />
          <p className="mt-4 text-gray-600">Loading VidIntel Pro...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeView={activeView} onViewChange={setActiveView} />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveView()}
      </main>
    </div>
  );
};

export default VidIntelPro;