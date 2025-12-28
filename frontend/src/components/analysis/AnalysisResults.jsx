import React, { useState } from 'react';
import { 
  Download, 
  Eye, 
  Share2, 
  Clock, 
  Play, 
  Pause, 
  ChevronDown, 
  ChevronUp,
  FileText,
  BarChart3,
  Copy,
  Check
} from 'lucide-react';
import { Card, Button, Badge } from '../common';

const AnalysisResults = ({ 
  analysisData, 
  onAnalyzeNewVideo, 
  onDownloadReport, 
  onShareResults 
}) => {
  const [expandedSegments, setExpandedSegments] = useState(new Set());
  const [copiedSegment, setCopiedSegment] = useState(null);
  const [viewMode, setViewMode] = useState('segments'); // 'segments' | 'summary' | 'statistics'

  if (!analysisData || !analysisData.results) {
    return (
      <Card className="text-center py-8">
        <div className="text-gray-500">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No analysis results available</p>
        </div>
      </Card>
    );
  }

  const { results } = analysisData;

  const toggleSegmentExpansion = (index) => {
    const newExpanded = new Set(expandedSegments);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSegments(newExpanded);
  };

  const copySegmentText = async (segment, index) => {
    const text = `${formatTimeRange(segment.startTime, segment.endTime)}: ${segment.combinedNarrative || segment.visualDescription || segment.audioDescription}`;
    
    try {
      await navigator.clipboard.writeText(text);
      setCopiedSegment(index);
      setTimeout(() => setCopiedSegment(null), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatTimeRange = (startTime, endTime) => {
    return `${formatDuration(startTime)} - ${formatDuration(endTime)}`;
  };

  const handleDownloadReport = () => {
    if (onDownloadReport) {
      onDownloadReport(analysisData);
    } else {
      // Default download implementation
      const reportContent = generateReportContent();
      const blob = new Blob([reportContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `video-analysis-${analysisData.sessionId || 'report'}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const handleShareResults = () => {
    if (onShareResults) {
      onShareResults(analysisData);
    } else {
      // Default share implementation
      const shareText = `Video Analysis Results:\n\n${generateReportContent()}`;
      if (navigator.share) {
        navigator.share({
          title: 'Video Analysis Results',
          text: shareText
        });
      } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(shareText);
        alert('Results copied to clipboard!');
      }
    }
  };

  const generateReportContent = () => {
    let content = `Video Analysis Report\n`;
    content += `========================\n\n`;
    content += `Video: ${analysisData.videoFile?.name || 'Unknown'}\n`;
    content += `Duration: ${formatDuration(results.duration || 0)}\n`;
    content += `Content Type: ${results.contentType || 'Unknown'}\n`;
    content += `Segments: ${results.segments?.length || 0}\n`;
    content += `Analysis Date: ${new Date(analysisData.completedAt || Date.now()).toLocaleDateString()}\n\n`;

    if (results.segments) {
      content += `Segment Analysis:\n`;
      content += `=================\n\n`;
      
      results.segments.forEach((segment, index) => {
        content += `${formatTimeRange(segment.startTime, segment.endTime)}:\n`;
        if (segment.combinedNarrative) {
          content += `${segment.combinedNarrative}\n\n`;
        } else {
          if (segment.visualDescription) {
            content += `Visual: ${segment.visualDescription}\n`;
          }
          if (segment.audioDescription) {
            content += `Audio: ${segment.audioDescription}\n`;
          }
          content += `\n`;
        }
      });
    }

    return content;
  };

  const calculateStatistics = () => {
    if (!results.segments) return {};

    const totalSegments = results.segments.length;
    const avgConfidence = results.segments.reduce((sum, seg) => sum + (seg.confidence || 0), 0) / totalSegments;
    const segmentsWithVisual = results.segments.filter(seg => seg.visualDescription).length;
    const segmentsWithAudio = results.segments.filter(seg => seg.audioDescription).length;
    const segmentsWithNarrative = results.segments.filter(seg => seg.combinedNarrative).length;

    return {
      totalSegments,
      avgConfidence: Math.round(avgConfidence * 100),
      visualCoverage: Math.round((segmentsWithVisual / totalSegments) * 100),
      audioCoverage: Math.round((segmentsWithAudio / totalSegments) * 100),
      narrativeCoverage: Math.round((segmentsWithNarrative / totalSegments) * 100)
    };
  };

  const statistics = calculateStatistics();

  return (
    <div className="space-y-6">
      {/* Header with Summary Stats */}
      <Card>
        <div className="space-y-6">
          {/* Title and Actions */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
              <p className="text-gray-600 mt-1">
                {analysisData.videoFile?.name || 'Unknown video'} • 
                Completed {new Date(analysisData.completedAt || Date.now()).toLocaleDateString()}
              </p>
            </div>
            <div className="flex space-x-3">
              <Button variant="outline" onClick={handleDownloadReport}>
                <Download className="h-4 w-4 mr-2" />
                Download Report
              </Button>
              <Button variant="outline" onClick={handleShareResults}>
                <Share2 className="h-4 w-4 mr-2" />
                Share Results
              </Button>
              {onAnalyzeNewVideo && (
                <Button onClick={onAnalyzeNewVideo}>
                  Analyze New Video
                </Button>
              )}
            </div>
          </div>

          {/* Summary Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {Math.round((results.duration || 0) / 60)}m
              </div>
              <div className="text-sm text-blue-600">Total Duration</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {results.segments?.length || 0}
              </div>
              <div className="text-sm text-green-600">Segments</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {results.contentType || 'Unknown'}
              </div>
              <div className="text-sm text-purple-600">Content Type</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {statistics.avgConfidence || 0}%
              </div>
              <div className="text-sm text-orange-600">Avg Confidence</div>
            </div>
          </div>

          {/* View Mode Selector */}
          <div className="flex space-x-2 border-b border-gray-200">
            <button
              onClick={() => setViewMode('segments')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                viewMode === 'segments'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <FileText className="h-4 w-4 inline mr-2" />
              Segments
            </button>
            <button
              onClick={() => setViewMode('summary')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                viewMode === 'summary'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Eye className="h-4 w-4 inline mr-2" />
              Summary
            </button>
            <button
              onClick={() => setViewMode('statistics')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                viewMode === 'statistics'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <BarChart3 className="h-4 w-4 inline mr-2" />
              Statistics
            </button>
          </div>
        </div>
      </Card>

      {/* Content based on view mode */}
      {viewMode === 'segments' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Time-Segmented Analysis</h3>
          {results.segments?.map((segment, index) => (
            <Card key={index} className="transition-all duration-200 hover:shadow-md">
              <div className="space-y-4">
                {/* Segment Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Clock className="h-5 w-5 text-gray-400" />
                    <h4 className="font-semibold text-gray-900">
                      {formatTimeRange(segment.startTime, segment.endTime)}
                    </h4>
                    {segment.confidence && (
                      <Badge variant="outline">
                        {Math.round(segment.confidence * 100)}% confidence
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copySegmentText(segment, index)}
                    >
                      {copiedSegment === index ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => toggleSegmentExpansion(index)}
                    >
                      {expandedSegments.has(index) ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>

                {/* Main Content */}
                {segment.combinedNarrative && (
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h5 className="text-sm font-medium text-gray-700 mb-2">Summary:</h5>
                    <p className="text-gray-800 leading-relaxed">{segment.combinedNarrative}</p>
                  </div>
                )}

                {/* Expanded Details */}
                {expandedSegments.has(index) && (
                  <div className="space-y-3 pt-2 border-t border-gray-100">
                    {segment.visualDescription && (
                      <div>
                        <h6 className="text-sm font-medium text-gray-700 mb-1">Visual Analysis:</h6>
                        <p className="text-sm text-gray-600 leading-relaxed">{segment.visualDescription}</p>
                      </div>
                    )}
                    
                    {segment.audioDescription && (
                      <div>
                        <h6 className="text-sm font-medium text-gray-700 mb-1">Audio Analysis:</h6>
                        <p className="text-sm text-gray-600 leading-relaxed">{segment.audioDescription}</p>
                      </div>
                    )}

                    {segment.transcription && (
                      <div>
                        <h6 className="text-sm font-medium text-gray-700 mb-1">Transcription:</h6>
                        <p className="text-sm text-gray-600 italic leading-relaxed">"{segment.transcription}"</p>
                      </div>
                    )}

                    {segment.detectedObjects && segment.detectedObjects.length > 0 && (
                      <div>
                        <h6 className="text-sm font-medium text-gray-700 mb-1">Detected Objects:</h6>
                        <div className="flex flex-wrap gap-1">
                          {segment.detectedObjects.map((obj, objIndex) => (
                            <Badge key={objIndex} variant="secondary" size="sm">
                              {obj.class_name} ({Math.round(obj.confidence * 100)}%)
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {viewMode === 'summary' && (
        <Card title="Analysis Summary">
          <div className="space-y-4">
            <div className="prose max-w-none">
              <p className="text-gray-700 leading-relaxed">
                This {results.contentType?.toLowerCase() || 'video'} analysis covers {Math.round((results.duration || 0) / 60)} minutes 
                of content across {results.segments?.length || 0} segments. The analysis provides detailed insights into both 
                visual and audio elements throughout the video.
              </p>
            </div>

            {/* Key Highlights */}
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">Key Highlights:</h4>
              <div className="space-y-2">
                {results.segments?.slice(0, 3).map((segment, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Clock className="h-4 w-4 text-gray-400 mt-1 flex-shrink-0" />
                    <div>
                      <span className="text-sm font-medium text-gray-700">
                        {formatTimeRange(segment.startTime, segment.endTime)}:
                      </span>
                      <p className="text-sm text-gray-600 mt-1">
                        {segment.combinedNarrative || segment.visualDescription || segment.audioDescription}
                      </p>
                    </div>
                  </div>
                ))}
                {results.segments?.length > 3 && (
                  <p className="text-sm text-gray-500 text-center py-2">
                    ... and {results.segments.length - 3} more segments
                  </p>
                )}
              </div>
            </div>
          </div>
        </Card>
      )}

      {viewMode === 'statistics' && (
        <Card title="Analysis Statistics">
          <div className="space-y-6">
            {/* Coverage Statistics */}
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">Analysis Coverage</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Visual Analysis</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${statistics.visualCoverage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12">
                      {statistics.visualCoverage}%
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Audio Analysis</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${statistics.audioCoverage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12">
                      {statistics.audioCoverage}%
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Combined Narrative</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full" 
                        style={{ width: `${statistics.narrativeCoverage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12">
                      {statistics.narrativeCoverage}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quality Metrics */}
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">Quality Metrics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-xl font-bold text-blue-600">{statistics.totalSegments}</div>
                  <div className="text-xs text-blue-600">Total Segments</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-xl font-bold text-green-600">{statistics.avgConfidence}%</div>
                  <div className="text-xs text-green-600">Avg Confidence</div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <div className="text-xl font-bold text-purple-600">
                    {Math.round((results.duration || 0) / (results.segments?.length || 1))}s
                  </div>
                  <div className="text-xs text-purple-600">Avg Segment</div>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <div className="text-xl font-bold text-orange-600">
                    {results.contentType || 'N/A'}
                  </div>
                  <div className="text-xs text-orange-600">Content Type</div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default AnalysisResults;