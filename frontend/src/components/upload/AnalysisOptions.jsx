import React from 'react';
import { Card } from '../common';

const AnalysisOptions = ({ options, onChange }) => {
  return (
    <Card title="Analysis Options">
      <div className="space-y-4">
        <div>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.transcription}
              onChange={(e) => onChange({ ...options, transcription: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Speech-to-Text Transcription</span>
          </label>
        </div>
        
        <div>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.sentiment}
              onChange={(e) => onChange({ ...options, sentiment: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Sentiment Analysis</span>
          </label>
        </div>
        
        <div>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.faceDetection}
              onChange={(e) => onChange({ ...options, faceDetection: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Face Detection</span>
          </label>
        </div>
        
        <div>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.sceneDetection}
              onChange={(e) => onChange({ ...options, sceneDetection: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Scene Change Detection</span>
          </label>
        </div>
        
        <div>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={options.contentModeration}
              onChange={(e) => onChange({ ...options, contentModeration: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Content Moderation</span>
          </label>
        </div>
      </div>
    </Card>
  );
};

export default AnalysisOptions;