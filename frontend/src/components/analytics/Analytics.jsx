import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { TrendingUp, CheckCircle, Brain } from 'lucide-react';
import { Card, Badge, LoadingSpinner } from '../common';
import MetricsChart from './MetricsChart';

const Analytics = ({ videoId }) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');

  useEffect(() => {
    if (!videoId) return;

    const fetchDashboardData = async () => {
      try {
        const res = await axios.get(`/analytics/dashboard/${videoId}`);
        console.log('Dashboard data:', res.data); // debug
        setDashboardData(res.data);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [videoId]);

  if (!videoId) {
    return (
      <div className="text-center text-red-600 font-semibold py-8">
        No video selected. Please choose a video to view analytics.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner />
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="text-center text-red-500 font-medium">
        Failed to load dashboard data.
      </div>
    );
  }

  const { video_info, analysis_summary } = dashboardData;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="24h">Last 24 hours</option>
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
        </select>
      </div>

      {video_info ? (
        <div className="text-sm text-gray-700 bg-gray-50 rounded-lg p-4 border">
          <p><strong>Filename:</strong> {video_info.filename}</p>
          <p><strong>Duration:</strong> {video_info.duration_minutes} mins</p>
          <p><strong>Resolution:</strong> {video_info.resolution} @ {video_info.fps} fps</p>
          <p><strong>Size:</strong> {video_info.file_size_mb} MB</p>
          <p><strong>Uploaded:</strong> {new Date(video_info.upload_time).toLocaleString()}</p>
        </div>
      ) : (
        <div className="text-red-500">Video metadata not available.</div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart title="Upload Trends" data={[]} type="line" />
        <MetricsChart
          title="Analysis Types"
          data={Object.keys(analysis_summary || {})}
          type="pie"
        />
        <MetricsChart title="Content Categories" data={[]} type="bar" />

        <Card title="Top Insights">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center">
                <TrendingUp className="h-5 w-5 text-blue-600 mr-3" />
                <div>
                  <p className="font-medium text-blue-900">Peak Upload Time</p>
                  <p className="text-sm text-blue-600">2:00 PM - 4:00 PM</p>
                </div>
              </div>
              <Badge variant="info">+24%</Badge>
            </div>

            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 text-green-600 mr-3" />
                <div>
                  <p className="font-medium text-green-900">Success Rate</p>
                  <p className="text-sm text-green-600">100%</p>
                </div>
              </div>
              <Badge variant="success">Stable</Badge>
            </div>

            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center">
                <Brain className="h-5 w-5 text-purple-600 mr-3" />
                <div>
                  <p className="font-medium text-purple-900">Top Feature</p>
                  <p className="text-sm text-purple-600">Transcription</p>
                </div>
              </div>
              <Badge variant="purple">Popular</Badge>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Analytics;
