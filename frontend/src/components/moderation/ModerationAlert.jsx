import React from 'react';
import { AlertTriangle, Eye, CheckCircle } from 'lucide-react';
import { Badge } from '../common';

const ModerationAlert = ({ severity, title, description, videoId, timestamp }) => {
  const severityConfig = {
    high: { color: 'danger', icon: AlertTriangle, bgColor: 'bg-red-50 border-red-200' },
    medium: { color: 'warning', icon: Eye, bgColor: 'bg-yellow-50 border-yellow-200' },
    low: { color: 'info', icon: CheckCircle, bgColor: 'bg-blue-50 border-blue-200' }
  };

  const config = severityConfig[severity] || severityConfig.low;
  const Icon = config.icon;

  return (
    <div className={`p-4 rounded-lg border ${config.bgColor}`}>
      <div className="flex items-start">
        <Icon className={`h-5 w-5 mt-0.5 mr-3 ${
          severity === 'high' ? 'text-red-600' : 
          severity === 'medium' ? 'text-yellow-600' : 'text-blue-600'
        }`} />
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900">{title}</h4>
            <Badge variant={config.color} size="sm">{severity.toUpperCase()}</Badge>
          </div>
          <p className="text-sm text-gray-600 mt-1">{description}</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-gray-500">Video ID: {videoId}</span>
            <span className="text-xs text-gray-500">{timestamp}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModerationAlert;