import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Pause, Download, Eye } from 'lucide-react';
import { Card, Button, Badge, ProgressBar } from '../common';
import LoadingSpinner from '../common/LoadingSpinner';
import AnalysisProgress from './AnalysisProgress';
import AnalysisResults from './AnalysisResults';
import { useAPI } from '../../hooks';

const AnalysisPage = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const api = useAPI();
  
  const [analysisData, setAnalysisData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [error, setError] = useState(null);
  const [preliminaryResults, setPreliminaryResults] = useState([]);

  useEffect(() => {
    // Load analysis data from session storage or start new analysis
    const loadAnalysisData = () => {
      try {
        const sessionData = sessionStorage.getItem(`analysis_${sessionId}`);
        if (sessionData) {
          const data = JSON.parse(sessionData);
          setAnalysisData(data);
          
          // If analysis is still in progress, continue monitoring
          if (data.status === 'analyzing') {
            setIsAnalyzing(true);
            startAnalysisMonitoring(data);
          }
        } else {
          setError('Analysis session not found');
        }
      } catch (err) {
        console.error('Failed to load analysis data:', err);
        setError('Failed to load analysis data');
      }
    };

    if (sessionId) {
      loadAnalysisData();
    }
  }, [sessionId]);

  const startAnalysisMonitoring = async (data) => {
    try {
      setIsAnalyzing(true);
      setCurrentTask('Processing video segments...');
      
      // Simulate real-time progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = Math.min(prev + Math.random() * 5, 95);
          return newProgress;
        });
      }, 2000);

      // Simulate task updates
      const tasks = [
        'Extracting video frames...',
        'Processing audio stream...',
        'Analyzing visual content...',
        'Generating descriptions...',
        'Compiling final report...'
      ];

      let taskIndex = 0;
      const taskInterval = setInterval(() => {
        if (taskIndex < tasks.length) {
          setCurrentTask(tasks[taskIndex]);
          taskIndex++;
        }
      }, 3000);

      // For now, simulate the analysis since we don't have the actual file
      // In a real implementation, the backend would handle the file processing
      setTimeout(async () => {
        clearInterval(progressInterval);
        clearInterval(taskInterval);
        
        setProgress(100);
        setCurrentTask('Analysis complete!');
        setIsAnalyzing(false);
        
        // Simulate analysis results
        const mockResults = {
          duration: 300, // 5 minutes
          contentType: 'Educational',
          segments: [
            {
              startTime: 0,
              endTime: 300,
              visualDescription: 'Video content showing educational material with clear visual elements',
              audioDescription: 'Clear narration explaining key concepts',
              combinedNarrative: '0-5 minutes: Educational content with clear narration and visual aids',
              confidence: 0.85
            }
          ]
        };
        
        // Update analysis data with results
        const updatedData = {
          ...data,
          status: 'completed',
          results: mockResults,
          completedAt: new Date().toISOString()
        };
        
        setAnalysisData(updatedData);
        
        // Update session storage
        sessionStorage.setItem(`analysis_${sessionId}`, JSON.stringify(updatedData));
      }, 10000); // Complete after 10 seconds for demo
      
    } catch (err) {
      console.error('Analysis failed:', err);
      setError('Analysis failed: ' + err.message);
      setIsAnalyzing(false);
      setCurrentTask('Analysis failed');
    }
  };

  const handleBackToUpload = () => {
    navigate('/upload');
  };

  const handleAnalyzeNewVideo = () => {
    // Clear current session and navigate to upload
    if (sessionId) {
      sessionStorage.removeItem(`analysis_${sessionId}`);
    }
    navigate('/upload');
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="max-w-md w-full">
          <div className="text-center space-y-4">
            <div className="text-red-500 text-xl">⚠️</div>
            <h2 className="text-xl font-semibold text-gray-900">Analysis Error</h2>
            <p className="text-gray-600">{error}</p>
            <Button onClick={handleBackToUpload} className="w-full">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Upload
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="xl" />
          <p className="mt-4 text-gray-600">Loading analysis...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button variant="outline" onClick={handleBackToUpload}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Upload
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Video Analysis</h1>
                <p className="text-gray-600">
                  {analysisData.videoFile?.name || 'Unknown video'}
                </p>
              </div>
            </div>
            {analysisData.status === 'completed' && (
              <Button onClick={handleAnalyzeNewVideo}>
                Analyze New Video
              </Button>
            )}
          </div>
        </div>

        {/* Analysis Progress */}
        {isAnalyzing && (
          <AnalysisProgress
            sessionId={sessionId}
            onProgressUpdate={(data) => {
              setProgress(data.progress || 0);
              setCurrentTask(data.current_segment || data.status || 'Processing...');
              if (data.preliminary_results) {
                setPreliminaryResults(data.preliminary_results);
              }
            }}
            onComplete={(data) => {
              setIsAnalyzing(false);
              setProgress(100);
              setCurrentTask('Analysis complete!');
              
              // Simulate final results for now
              const mockResults = {
                duration: 300,
                contentType: 'Educational',
                segments: data.preliminary_results || [
                  {
                    startTime: 0,
                    endTime: 300,
                    visualDescription: 'Video content showing educational material with clear visual elements',
                    audioDescription: 'Clear narration explaining key concepts',
                    combinedNarrative: '0-5 minutes: Educational content with clear narration and visual aids',
                    confidence: 0.85
                  }
                ]
              };
              
              const updatedData = {
                ...analysisData,
                status: 'completed',
                results: mockResults,
                completedAt: new Date().toISOString()
              };
              
              setAnalysisData(updatedData);
              sessionStorage.setItem(`analysis_${sessionId}`, JSON.stringify(updatedData));
            }}
            onError={(error) => {
              setError(error);
              setIsAnalyzing(false);
            }}
          />
        )}

        {/* Preliminary Results */}
        {preliminaryResults.length > 0 && (
          <Card className="mb-8" title="Preliminary Results">
            <div className="space-y-4">
              {preliminaryResults.map((result, index) => (
                <div key={index} className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">
                      Segment {index + 1} ({formatDuration(result.startTime)} - {formatDuration(result.endTime)})
                    </span>
                    <Badge variant="success">Complete</Badge>
                  </div>
                  <p className="text-sm text-gray-600">{result.description}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Final Results */}
        {analysisData.status === 'completed' && analysisData.results && (
          <AnalysisResults
            analysisData={analysisData}
            onAnalyzeNewVideo={handleAnalyzeNewVideo}
            onDownloadReport={(data) => {
              // Custom download handler if needed
              console.log('Downloading report for:', data.sessionId);
            }}
            onShareResults={(data) => {
              // Custom share handler if needed
              console.log('Sharing results for:', data.sessionId);
            }}
          />
        )}
      </div>
    </div>
  );
};

export default AnalysisPage;