import React, { useState } from 'react';
import { Download, Eye } from 'lucide-react';
import { jsPDF } from 'jspdf';
import { Card, Button, Badge } from '../common';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../common/Dialog';
import FileUpload from './FileUpload';
import AnalysisOptions from './AnalysisOptions';
import AnalysisProgress from './AnalysisProgress';
import { useAPI } from '../../hooks';

const UploadSection = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [analysisOptions, setAnalysisOptions] = useState({
    transcription: true,
    sentiment: true,
    faceDetection: false,
    sceneDetection: true,
    contentModeration: true,
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const api = useAPI();

  const handleAnalyze = async () => {
    if (!selectedFiles.length) return;

    setIsAnalyzing(true);
    setProgress(0);
    setCurrentTask('Preparing analysis...');

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? 90 : prev + Math.random() * 10));
      }, 1000);

      const tasks = [
        'Extracting audio...',
        'Generating transcript...',
        'Analyzing sentiment...',
        'Processing video features...',
        'Finalizing results...',
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

      const results = await api.batchAnalyze(selectedFiles);
      clearInterval(progressInterval);
      clearInterval(taskInterval);
      setProgress(100);
      setCurrentTask('Analysis complete!');
      setAnalysisResults(results.batch_results || []);

      setTimeout(() => {
        setIsAnalyzing(false);
      }, 1000);
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
      setCurrentTask('Analysis failed');
    }
  };

  const handleDownloadPDF = (result, index) => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text('Video Analysis Report', 20, 20);

    doc.setFontSize(12);
    doc.text(`Duration: ${Math.round(result.metadata?.duration / 60) || 0} minutes`, 20, 40);
    doc.text(`Sentiment: ${result.sentiment_analysis?.label || 'N/A'}`, 20, 50);
    doc.text('Transcript Preview:', 20, 65);
    doc.setFontSize(10);
    doc.text(result.transcript?.substring(0, 500) || 'No transcript available', 20, 75, { maxWidth: 170 });

    doc.save(`video_analysis_report_${index + 1}.pdf`);
  };

  const handleViewDetails = (result) => {
    setSelectedResult(result);
    setDetailsOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card title="Upload Video">
            <FileUpload onFileSelect={setSelectedFiles} />
            {selectedFiles.length > 0 && (
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

          <AnalysisOptions options={analysisOptions} onChange={setAnalysisOptions} />
        </div>

        <div className="space-y-6">
          <AnalysisProgress isAnalyzing={isAnalyzing} progress={progress} currentTask={currentTask} />

          {analysisResults && (
            <Card title="Analysis Results">
              <div className="space-y-6">
                {Array.isArray(analysisResults) && analysisResults.length > 0 ? (
                  analysisResults.map((result, idx) => (
                    <div
                      key={idx}
                      className="p-4 border rounded-lg bg-white shadow-sm space-y-4"
                    >
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-4 bg-blue-50 rounded-lg">
                          <p className="text-2xl font-bold text-blue-600">
                            {Math.round(result.metadata?.duration / 60) || 0}m
                          </p>
                          <p className="text-sm text-blue-600">Duration</p>
                        </div>

                        <div className="text-center p-4 bg-green-50 rounded-lg">
                          <Badge
                            variant={
                              result.sentiment_analysis?.label === 'Positive'
                                ? 'success'
                                : result.sentiment_analysis?.label === 'Negative'
                                ? 'danger'
                                : 'info'
                            }
                            size="lg"
                          >
                            {result.sentiment_analysis?.label || 'N/A'}
                          </Badge>
                          <p className="text-sm text-green-600">Sentiment</p>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Transcript Preview</h4>
                        <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                          {result.transcript?.substring(0, 200) || 'No transcript available'}...
                        </p>
                      </div>

                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline" onClick={() => handleDownloadPDF(result, idx)}>
                          <Download className="h-4 w-4 mr-2" />
                          Download Report
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => handleViewDetails(result)}>
                          <Eye className="h-4 w-4 mr-2" />
                          View Details
                        </Button>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500">No results available.</p>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>

      {selectedResult && (
        <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Detailed Analysis</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto">
              <p><strong>Duration:</strong> {Math.round(selectedResult.metadata?.duration / 60)} minutes</p>
              <p><strong>Sentiment:</strong> {selectedResult.sentiment_analysis?.label}</p>
              <div>
                <strong>Transcript:</strong>
                <div className="bg-gray-100 p-3 mt-1 rounded text-sm text-gray-800 whitespace-pre-wrap">
                  {selectedResult.transcript || 'No transcript available'}
                </div>
              </div>
              {/* Add more detailed sections here as needed */}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default UploadSection;
