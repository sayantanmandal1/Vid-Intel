import React from 'react';
import { Card, LoadingSpinner, ProgressBar } from '../common';

const AnalysisProgress = ({ isAnalyzing, progress, currentTask }) => {
  if (!isAnalyzing) return null;

  return (
    <Card title="Analysis in Progress">
      <div className="space-y-4">
        <div className="flex items-center">
          <LoadingSpinner size="sm" />
          <span className="ml-2 text-sm text-gray-600">{currentTask || 'Processing...'}</span>
        </div>
        
        <ProgressBar value={progress} showLabel />
        
        <div className="text-xs text-gray-500">
          This may take a few minutes depending on video length
        </div>
      </div>
    </Card>
  );
};

export default AnalysisProgress;