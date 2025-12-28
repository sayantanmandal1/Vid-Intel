// VidIntelPro.jsx
import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import Header from './components/layout/Header';
import Dashboard from './components/dashboard/Dashboard';
import UploadSection from './components/upload/UploadSection';
//import Analytics from './components/analytics/Analytics';
import Settings from './components/settings/Settings';
import Moderation from './components/moderation/Moderation';
import LoadingSpinner from './components/common/LoadingSpinner';
import { useAPI } from './hooks/useAPI';
import { Notifications } from './components';
import AnalyticsWrapper from './components/analytics/AnalyticsWrapper';
import { AnalysisPage } from './components/analysis';


const VidIntelPro = () => {
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
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" />} />
            <Route path="/dashboard" element={<Dashboard platformStats={platformStats} />} />
            <Route path="/upload" element={<UploadSection />} />
            <Route path="/analysis/:sessionId" element={<AnalysisPage />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/notifications" element={<Notifications />} />
            <Route path="/analytics/:videoId" element={<AnalyticsWrapper />} />
            <Route path="/moderation" element={<Moderation />} />
            <Route path="*" element={<h2>404 - Page not found</h2>} />
          </Routes>

        </main>
      </div>
    </Router>
  );
};

export default VidIntelPro;
