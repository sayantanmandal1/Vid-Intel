import React from 'react';
import { Video, Clock, Brain, Activity, Upload, BarChart3, Shield, Search } from 'lucide-react';
import StatsCard from './StatsCard';
import RecentActivity from './RecentActivity';
import Card from '../common/Card';
import Button from '../common/Button';
import { useNavigate } from 'react-router-dom';

const Dashboard = ({ platformStats }) => {
  const navigate = useNavigate(); 
  const stats = [
    {
      title: 'Total Videos',
      value: platformStats?.platform_overview?.total_videos_processed || '0',
      change: '+12%',
      icon: Video,
      color: 'blue'
    },
    {
      title: 'Content Hours',
      value: `${platformStats?.platform_overview?.total_content_hours || '0'}h`,
      change: '+8%',
      icon: Clock,
      color: 'green'
    },
    {
      title: 'Analyses Done',
      value: platformStats?.analysis_statistics?.total_analyses_performed || '0',
      change: '+23%',
      icon: Brain,
      color: 'purple'
    },
    {
      title: 'Storage Used',
      value: `${platformStats?.platform_overview?.total_storage_gb || '0'}GB`,
      change: '+5%',
      icon: Activity,
      color: 'orange'
    }
  ];

  const recentActivities = [
    { description: 'Video analysis completed', timestamp: '2 minutes ago' },
    { description: 'New video uploaded', timestamp: '5 minutes ago' },
    { description: 'Content moderation check passed', timestamp: '10 minutes ago' },
    { description: 'Engagement metrics generated', timestamp: '15 minutes ago' }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentActivity activities={recentActivities} />
        
        <Card title="Quick Actions">
            <div className="grid grid-cols-2 gap-4">
                <Button
                variant="outline"
                size="lg"
                className="h-20 flex-col"
                onClick={() => navigate('/upload')}
                >
                <Upload className="h-6 w-6 mb-2" />
                Upload Video
                </Button>

                <Button
                variant="outline"
                size="lg"
                className="h-20 flex-col"
                onClick={() => navigate('/analytics')}
                >
                <BarChart3 className="h-6 w-6 mb-2" />
                View Analytics
                </Button>

                <Button
                variant="outline"
                size="lg"
                className="h-20 flex-col"
                onClick={() => navigate('/moderation')}
                >
                <Shield className="h-6 w-6 mb-2" />
                Moderation
                </Button>

                <Button
                variant="outline"
                size="lg"
                className="h-20 flex-col"
                onClick={() => navigate('/search')}
                >
                <Search className="h-6 w-6 mb-2" />
                Search Videos
                </Button>
            </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;