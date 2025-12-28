import React, { useState, useEffect, useRef } from 'react';
import { Clock, Activity, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { Card, Badge, ProgressBar, LoadingSpinner } from '../common';

const AnalysisProgress = ({ sessionId, onProgressUpdate, onComplete, onError }) => {
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('Initializing analysis...');
  const [segmentsCompleted, setSegmentsCompleted] = useState(0);
  const [totalSegments, setTotalSegments] = useState(0);
  const [preliminaryResults, setPreliminaryResults] = useState([]);
  const [estimatedCompletion, setEstimatedCompletion] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const maxConnectionAttempts = 5;
  
  const wsRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const startTimeRef = useRef(Date.now());

  useEffect(() => {
    if (!sessionId) return;

    // Try WebSocket connection first, fallback to polling
    const connectWebSocket = () => {
      if (connectionAttempts >= maxConnectionAttempts) {
        console.log('Max WebSocket connection attempts reached, switching to polling');
        startPolling();
        return;
      }

      try {
        const wsUrl = `ws://localhost:8000/ws/analysis/${sessionId}`;
        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = () => {
          console.log('WebSocket connected for analysis progress');
          setIsConnected(true);
          setError(null);
          setConnectionAttempts(0); // Reset on successful connection
        };

        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleProgressUpdate(data);
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
          }
        };

        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
          setConnectionAttempts(prev => prev + 1);
        };

        wsRef.current.onclose = (event) => {
          console.log('WebSocket connection closed', event.code, event.reason);
          setIsConnected(false);
          
          // Only reconnect if analysis is still in progress and we haven't had too many failures
          if (progress < 100 && !error && event.code !== 1000 && connectionAttempts < maxConnectionAttempts) {
            // Wait longer between reconnection attempts to avoid spam
            setTimeout(() => {
              if (progress < 100) {
                connectWebSocket();
              }
            }, 5000); // 5 second delay
          } else {
            // Start polling as fallback
            console.log('Switching to polling mode');
            startPolling();
          }
        };
      } catch (err) {
        console.error('Failed to create WebSocket connection:', err);
        setConnectionAttempts(prev => prev + 1);
        startPolling();
      }
    };

    const startPolling = () => {
      if (pollIntervalRef.current) return; // Already polling

      pollIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8000/analyze/status/${sessionId}`);
          if (response.ok) {
            const data = await response.json();
            handleProgressUpdate(data);
          } else {
            console.error('Failed to fetch analysis status:', response.statusText);
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 2000); // Poll every 2 seconds
    };

    const handleProgressUpdate = (data) => {
      console.log('📊 Progress update received:', data);
      
      setProgress(data.progress || 0);
      setCurrentTask(data.current_segment || data.status || 'Processing...');
      setSegmentsCompleted(data.segments_completed || 0);
      setTotalSegments(data.total_segments || 0);
      
      console.log(`   Progress: ${data.progress || 0}%`);
      console.log(`   Task: ${data.current_segment || data.status || 'Processing...'}`);
      console.log(`   Segments: ${data.segments_completed || 0}/${data.total_segments || 0}`);
      
      // Calculate estimated completion time
      if (data.progress > 0 && data.progress < 100) {
        const elapsed = Date.now() - startTimeRef.current;
        const estimatedTotal = (elapsed / data.progress) * 100;
        const remaining = estimatedTotal - elapsed;
        setEstimatedCompletion(new Date(Date.now() + remaining));
        console.log(`   Estimated completion: ${new Date(Date.now() + remaining).toLocaleTimeString()}`);
      }

      // Handle preliminary results
      if (data.preliminary_results) {
        console.log(`📋 Preliminary results: ${data.preliminary_results.length} items`);
        setPreliminaryResults(prev => {
          const newResults = [...prev];
          data.preliminary_results.forEach(result => {
            const existingIndex = newResults.findIndex(r => r.segment_id === result.segment_id);
            if (existingIndex >= 0) {
              newResults[existingIndex] = result;
            } else {
              newResults.push(result);
            }
          });
          return newResults;
        });
      }

      // Handle completion
      if (data.status === 'completed' || data.progress >= 100) {
        console.log('✅ Analysis completed!');
        setProgress(100);
        setCurrentTask('Analysis complete!');
        if (onComplete) {
          onComplete(data);
        }
        cleanup();
      }

      // Handle errors
      if (data.status === 'failed' || data.error) {
        console.error('❌ Analysis failed:', data.error || 'Unknown error');
        setError(data.error || 'Analysis failed');
        if (onError) {
          onError(data.error || 'Analysis failed');
        }
        cleanup();
      }

      // Notify parent component
      if (onProgressUpdate) {
        onProgressUpdate(data);
      }
    };

    const cleanup = () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };

    // Start connection
    connectWebSocket();

    // Cleanup on unmount
    return cleanup;
  }, [sessionId, onProgressUpdate, onComplete, onError]);

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatEstimatedTime = (date) => {
    if (!date) return 'Calculating...';
    const now = new Date();
    const diff = Math.max(0, date.getTime() - now.getTime());
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    if (minutes > 0) {
      return `~${minutes}m ${seconds}s remaining`;
    }
    return `~${seconds}s remaining`;
  };

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <div className="flex items-center space-x-3">
          <AlertCircle className="h-6 w-6 text-red-500" />
          <div>
            <h3 className="text-lg font-semibold text-red-800">Analysis Error</h3>
            <p className="text-red-600">{error}</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Progress Card */}
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <span>Analysis in Progress</span>
            </h3>
            <div className="flex items-center space-x-2">
              <Badge variant={isConnected ? "success" : "warning"}>
                {isConnected ? "Live Updates" : "Polling"}
              </Badge>
              <Badge variant="info">
                {Math.round(progress)}%
              </Badge>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <ProgressBar value={progress} max={100} className="h-3" />
            <div className="flex justify-between text-sm text-gray-600">
              <span>{currentTask}</span>
              <span>{Math.round(progress)}% complete</span>
            </div>
          </div>

          {/* Segment Progress */}
          {totalSegments > 0 && (
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">
                  Segments: {segmentsCompleted} / {totalSegments}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                {formatEstimatedTime(estimatedCompletion)}
              </div>
            </div>
          )}

          {/* Current Task Status */}
          <div className="flex items-center space-x-2 p-3 bg-blue-50 rounded-lg">
            <LoadingSpinner size="sm" />
            <span className="text-sm text-blue-800">{currentTask}</span>
          </div>
        </div>
      </Card>

      {/* Preliminary Results */}
      {preliminaryResults.length > 0 && (
        <Card title="Preliminary Results">
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              <Clock className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600">
                Results appear as segments are processed
              </span>
            </div>
            
            {preliminaryResults.map((result, index) => (
              <div key={result.segment_id || index} className="p-4 bg-gray-50 rounded-lg border-l-4 border-green-400">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">
                    Segment {index + 1} ({formatDuration(result.start_time || 0)} - {formatDuration(result.end_time || 0)})
                  </span>
                  <Badge variant="success" size="sm">
                    Complete
                  </Badge>
                </div>
                
                {result.visual_description && (
                  <div className="mb-2">
                    <span className="text-xs font-medium text-gray-700">Visual:</span>
                    <p className="text-sm text-gray-600 mt-1">{result.visual_description}</p>
                  </div>
                )}
                
                {result.audio_description && (
                  <div className="mb-2">
                    <span className="text-xs font-medium text-gray-700">Audio:</span>
                    <p className="text-sm text-gray-600 mt-1">{result.audio_description}</p>
                  </div>
                )}
                
                {result.combined_narrative && (
                  <div>
                    <span className="text-xs font-medium text-gray-700">Summary:</span>
                    <p className="text-sm text-gray-800 font-medium mt-1">{result.combined_narrative}</p>
                  </div>
                )}
                
                {result.confidence && (
                  <div className="mt-2 flex justify-end">
                    <Badge variant="outline" size="sm">
                      Confidence: {Math.round(result.confidence * 100)}%
                    </Badge>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Processing Statistics */}
      <Card title="Processing Statistics">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{Math.round(progress)}%</div>
            <div className="text-xs text-blue-600">Progress</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{segmentsCompleted}</div>
            <div className="text-xs text-green-600">Completed</div>
          </div>
          <div className="text-center p-3 bg-yellow-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{totalSegments - segmentsCompleted}</div>
            <div className="text-xs text-yellow-600">Remaining</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{preliminaryResults.length}</div>
            <div className="text-xs text-purple-600">Results</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AnalysisProgress;