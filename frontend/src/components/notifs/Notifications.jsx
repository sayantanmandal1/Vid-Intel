import React from 'react';
import { Bell, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { Card, Badge, Button } from '../common';

const notifications = [
  {
    id: 1,
    title: 'Video Analysis Complete',
    description: 'Your uploaded video has finished processing.',
    timestamp: '2 hours ago',
    type: 'success',
    icon: CheckCircle,
  },
  {
    id: 2,
    title: 'Content Moderation Alert',
    description: 'Potentially sensitive content detected in your latest video.',
    timestamp: '6 hours ago',
    type: 'warning',
    icon: AlertTriangle,
  },
  {
    id: 3,
    title: 'System Update',
    description: 'New features have been added to your dashboard.',
    timestamp: '1 day ago',
    type: 'info',
    icon: Info,
  },
  {
    id: 4,
    title: 'Storage Limit Approaching',
    description: 'You have used 85% of your allocated storage.',
    timestamp: '2 days ago',
    type: 'danger',
    icon: AlertTriangle,
  },
];

const Notifications = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-gray-800 flex items-center">
          <Bell className="h-6 w-6 mr-2 text-blue-600" />
          Notifications
        </h2>
        <Button variant="outline" size="sm">
          Clear All
        </Button>
      </div>

      <Card>
        <ul className="divide-y divide-gray-200">
          {notifications.map(({ id, title, description, timestamp, type, icon: Icon }) => (
            <li key={id} className="py-4 flex items-start justify-between space-x-4">
              <div className="flex items-start space-x-3">
                <span className="mt-1">
                  <Icon className={`h-5 w-5 ${getIconColor(type)}`} />
                </span>
                <div>
                  <h3 className="font-medium text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-600">{description}</p>
                </div>
              </div>
              <div className="flex flex-col items-end space-y-1">
                <Badge variant={type} size="sm">
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </Badge>
                <span className="text-xs text-gray-400">{timestamp}</span>
              </div>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
};

const getIconColor = (type) => {
  switch (type) {
    case 'success':
      return 'text-green-600';
    case 'warning':
      return 'text-yellow-600';
    case 'danger':
      return 'text-red-600';
    case 'info':
      return 'text-blue-600';
    default:
      return 'text-gray-400';
  }
};

export default Notifications;
