import React, { useState } from 'react';
import { Download, Eye } from 'lucide-react';
import { Card, Button, Badge } from '../common';
import { useAPI } from '../../hooks';
import FileUpload from './FileUpload';
import AnalysisOptions from './AnalysisOptions';
import AnalysisProgress from './AnalysisProgress';

const UploadSection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    transcription: true,
    sentiment: true,
    faceDetection: false,
    sceneDetection: true,
    contentModeration: true
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  
  const api = useAPI();

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setProgress(0);
    setCurrentTask('Preparing analysis...');
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 1000);

      const tasks = [
        'Extracting audio...',
        'Generating transcript...',
        'Analyzing sentiment...',
        'Processing video features...',
        'Finalizing results...'
      ];
      
      let taskIndex = 0;
      const taskInterval = setInterval(() => {
        if (taskIndex < tasks.length) {
          setCurrentTask(tasks[taskIndex]);
          taskIndex++;
        } else {
          clearInterval(taskInterval);
        }
      }, 2000);

      const result = await api.uploadVideo(selectedFile);
      
      clearInterval(progressInterval);
      clearInterval(taskInterval);
      setProgress(100);
      setCurrentTask('Analysis complete!');
      setAnalysisResult(result);
      
      setTimeout(() => {
        setIsAnalyzing(false);
      }, 1000);
      
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
      setCurrentTask('Analysis failed');
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card title="Upload Video">
            <FileUpload onFileSelect={setSelectedFile} />
            
            {selectedFile && (
              <div className="mt-6">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  loading={isAnalyzing}
                  size="lg"
                  className="w-full"
                >
                  Start Analysis
                </Button>
              </div>
            )}
          </Card>
          
          <AnalysisOptions 
            options={analysisOptions}
            onChange={setAnalysisOptions}
          />
        </div>
        
        <div className="space-y-6">
          <AnalysisProgress
            isAnalyzing={isAnalyzing}
            progress={progress}
            currentTask={currentTask}
          />
          
          {analysisResult && (
            <Card title="Analysis Results">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <p className="text-2xl font-bold text-blue-600">
                      {Math.round(analysisResult.metadata?.duration / 60) || 0}m
                    </p>
                    <p className="text-sm text-blue-600">Duration</p>
                  </div>
                  
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <p className="text-2xl font-bold text-green-600">
                      {analysisResult.sentiment_analysis?.label || 'N/A'}
                    </p>
                    <p className="text-sm text-green-600">Sentiment</p>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h4 className="font-medium text-gray-900 mb-2">Transcript Preview</h4>
                  <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                    {analysisResult.transcript?.substring(0, 200) || 'No transcript available'}...
                  </p>
                </div>
                
                <div className="flex space-x-2">
                  <Button size="sm" variant="outline">
                    <Download className="h-4 w-4 mr-2" />
                    Download Report
                  </Button>
                  <Button size="sm" variant="outline">
                    <Eye className="h-4 w-4 mr-2" />
                    View Details
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadSection;