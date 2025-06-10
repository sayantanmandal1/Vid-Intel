import React, { useState } from 'react';
import { TrendingUp, CheckCircle, Brain } from 'lucide-react';
import { Card, Badge } from '../common';
import MetricsChart from './MetricsChart';

const Analytics = ({ platformStats }) => {
  const [timeRange, setTimeRange] = useState('7d');

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
        <div className="flex space-x-2">
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
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart 
          title="Upload Trends"
          data={[]}
          type="line"
        />
        
        <MetricsChart 
          title="Analysis Types"
          data={[]}
          type="pie"
        />
        
        <MetricsChart 
          title="Content Categories"
          data={[]}
          type="bar"
        />
        
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
                  <p className="font-medium text-green-900">Processing Success Rate</p>
                  <p className="text-sm text-green-600">98.5%</p>
                </div>
              </div>
              <Badge variant="success">Excellent</Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center">
                <Brain className="h-5 w-5 text-purple-600 mr-3" />
                <div>
                  <p className="font-medium text-purple-900">Most Used Feature</p>
                  <p className="text-sm text-purple-600">Sentiment Analysis</p>
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