import React, { useState } from 'react';
import { Settings, Eye, AlertTriangle, CheckCircle, Shield } from 'lucide-react';
import { Card, Button } from '../common';
import { StatsCard } from '../dashboard';
import ModerationAlert from './ModerationAlert';
import ContentFilter from './ContentFilter';

const Moderation = () => {
  const [filters, setFilters] = useState({ severity: 'all', category: 'all' });
  const [loading, setLoading] = useState(false);

  // Mock moderation alerts
  const mockAlerts = [
    {
      id: 1,
      severity: 'high',
      title: 'Inappropriate Content Detected',
      description: 'Adult content detected at timestamp 2:45-3:12',
      videoId: 'VID_001',
      timestamp: '5 minutes ago',
      category: 'content'
    },
    {
      id: 2,
      severity: 'medium',
      title: 'Profanity in Audio',
      description: 'Strong language detected in speech transcription',
      videoId: 'VID_002', 
      timestamp: '12 minutes ago',
      category: 'language'
    },
    {
      id: 3,
      severity: 'low',
      title: 'Copyright Music',
      description: 'Potential copyrighted music detected',
      videoId: 'VID_003',
      timestamp: '1 hour ago',
      category: 'copyright'
    }
  ];

  const [alerts, setAlerts] = useState(mockAlerts);

  const filteredAlerts = alerts.filter(alert => {
    if (filters.severity !== 'all' && alert.severity !== filters.severity) return false;
    if (filters.category !== 'all' && alert.category !== filters.category) return false;
    return true;
  });

  const handleModerationAction = async (alertId, action) => {
    setLoading(true);
    try {
      // Simulate API call
      setTimeout(() => {
        setAlerts(prev => prev.filter(alert => alert.id !== alertId));
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Moderation action failed:', error);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Content Moderation</h2>
        <Button>
          <Settings className="h-4 w-4 mr-2" />
          Moderation Settings
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard
          title="Pending Reviews"
          value="23"
          icon={Eye}
          color="orange"
        />
        <StatsCard
          title="High Priority"
          value="3"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Resolved Today"
          value="47"
          icon={CheckCircle}
          color="green"
        />
        <StatsCard
          title="Success Rate"
          value="94%"
          icon={Shield}
          color="blue"
        />
      </div>

      <Card 
        title="Content Alerts"
        actions={<ContentFilter filters={filters} onFilterChange={setFilters} />}
      >
        <div className="space-y-4">
          {filteredAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Shield className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No moderation alerts found</p>
            </div>
          ) : (
            filteredAlerts.map((alert) => (
              <div key={alert.id}>
                <ModerationAlert {...alert} />
                <div className="flex justify-end space-x-2 mt-2">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => handleModerationAction(alert.id, 'dismiss')}
                    disabled={loading}
                  >
                    Dismiss
                  </Button>
                  <Button 
                    size="sm" 
                    variant="danger"
                    onClick={() => handleModerationAction(alert.id, 'remove')}
                    disabled={loading}
                  >
                    Take Action
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
};

export default Moderation;