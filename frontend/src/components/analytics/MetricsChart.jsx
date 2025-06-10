import React from 'react';
import { BarChart3 } from 'lucide-react';
import { Card } from '../common';

const MetricsChart = ({ title, data, type = 'line' }) => {
  return (
    <Card title={title}>
      <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Chart visualization would be rendered here</p>
          <p className="text-sm">Using recharts or similar library</p>
        </div>
      </div>
    </Card>
  );
};

export default MetricsChart;