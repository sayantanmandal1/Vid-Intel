import React, { useState, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Play, 
  Pause, 
  BarChart3, 
  Brain, 
  Users, 
  Shield, 
  Zap,
  FileText,
  Download,
  Search,
  Settings,
  ChevronRight,
  Activity,
  Globe,
  Eye,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader,
  Video,
  Music,
  MessageSquare,
  Star,
  Filter
} from 'lucide-react';

// API Service Layer
class APIService {
  constructor(baseURL = 'http://127.0.0.1:8000') {
    this.baseURL = baseURL;
  }

  async uploadVideo(file, analysisType = 'comprehensive') {
    // Mock response for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          id: 'video_' + Date.now(),
          metadata: { duration: 300 },
          sentiment_analysis: { label: 'Positive' },
          transcript: 'This is a sample transcript of the video content...'
        });
      }, 3000);
    });
  }

  async getPlatformStats() {
    // Mock response for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          platform_overview: {
            total_videos_processed: 1247,
            total_content_hours: 856,
            total_storage_gb: 45.8
          },
          analysis_statistics: {
            total_analyses_performed: 3421
          }
        });
      }, 1000);
    });
  }
}

// Custom Hooks
const useAPI = () => {
  const [api] = useState(() => new APIService());
  return api;
};

// UI Components
const LoadingSpinner = ({ size = 'md', className = '' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <Loader className={`${sizeClasses[size]} animate-spin text-blue-600`} />
    </div>
  );
};

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  disabled = false, 
  loading = false,
  onClick,
  className = '',
  ...props 
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variants = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 shadow-sm',
    secondary: 'bg-gray-100 hover:bg-gray-200 text-gray-900 focus:ring-gray-500',
    outline: 'border border-gray-300 hover:bg-gray-50 text-gray-700 focus:ring-blue-500',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500',
    success: 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500'
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
    xl: 'px-8 py-4 text-lg'
  };

  return (
    <button
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${disabled || loading ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading && <LoadingSpinner size="sm" className="mr-2" />}
      {children}
    </button>
  );
};

const Card = ({ children, className = '', title, subtitle, actions }) => {
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 ${className}`}>
      {(title || subtitle || actions) && (
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
            {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
          </div>
          {actions && <div className="flex space-x-2">{actions}</div>}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

const Badge = ({ children, variant = 'default', size = 'md' }) => {
  const variants = {
    default: 'bg-gray-100 text-gray-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    danger: 'bg-red-100 text-red-800',
    info: 'bg-blue-100 text-blue-800',
    purple: 'bg-purple-100 text-purple-800'
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  };

  return (
    <span className={`inline-flex items-center font-medium rounded-full ${variants[variant]} ${sizes[size]}`}>
      {children}
    </span>
  );
};

const ProgressBar = ({ value, max = 100, className = '', showLabel = false }) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  return (
    <div className={`w-full ${className}`}>
      <div className="bg-gray-200 rounded-full h-2">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <div className="text-sm text-gray-600 mt-1">
          {Math.round(percentage)}%
        </div>
      )}
    </div>
  );
};

// Header Component
const Header = ({ activeView, onViewChange }) => {
  const navigation = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'analytics', label: 'Analytics', icon: Activity },
    { id: 'moderation', label: 'Moderation', icon: Shield }
  ];

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Brain className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">VidIntel Pro</span>
            </div>
          </div>
          
          <nav className="hidden md:flex space-x-8">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => onViewChange(item.id)}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeView === item.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="h-4 w-4 mr-2" />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center space-x-4">
            <button className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

// Dashboard Components
const StatsCard = ({ title, value, change, icon: Icon, color = 'blue' }) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    purple: 'text-purple-600 bg-purple-100',
    orange: 'text-orange-600 bg-orange-100',
    red: 'text-red-600 bg-red-100'
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <p className={`text-sm ${change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
              {change} from last month
            </p>
          )}
        </div>
      </div>
    </Card>
  );
};

const RecentActivity = ({ activities = [] }) => {
  return (
    <Card title="Recent Activity" className="h-full">
      <div className="space-y-4">
        {activities.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No recent activity</p>
          </div>
        ) : (
          activities.map((activity, index) => (
            <div key={index} className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-gray-900">{activity.description}</p>
                <p className="text-xs text-gray-500">{activity.timestamp}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};

const Dashboard = ({ platformStats }) => {
  const stats = [
    {
      title: 'Total Videos',
      value: platformStats?.platform_overview?.total_videos_processed || '0',
      change: '+12%',
      icon: Video,
      color: 'blue'
    },
    {
      title: 'Content Hours',
      value: `${platformStats?.platform_overview?.total_content_hours || '0'}h`,
      change: '+8%',
      icon: Clock,
      color: 'green'
    },
    {
      title: 'Analyses Done',
      value: platformStats?.analysis_statistics?.total_analyses_performed || '0',
      change: '+23%',
      icon: Brain,
      color: 'purple'
    },
    {
      title: 'Storage Used',
      value: `${platformStats?.platform_overview?.total_storage_gb || '0'}GB`,
      change: '+5%',
      icon: Activity,
      color: 'orange'
    }
  ];

  const recentActivities = [
    { description: 'Video analysis completed', timestamp: '2 minutes ago' },
    { description: 'New video uploaded', timestamp: '5 minutes ago' },
    { description: 'Content moderation check passed', timestamp: '10 minutes ago' },
    { description: 'Engagement metrics generated', timestamp: '15 minutes ago' }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentActivity activities={recentActivities} />
        
        <Card title="Quick Actions">
          <div className="grid grid-cols-2 gap-4">
            <Button variant="outline" size="lg" className="h-20 flex-col">
              <Upload className="h-6 w-6 mb-2" />
              Upload Video
            </Button>
            <Button variant="outline" size="lg" className="h-20 flex-col">
              <BarChart3 className="h-6 w-6 mb-2" />
              View Analytics
            </Button>
            <Button variant="outline" size="lg" className="h-20 flex-col">
              <Shield className="h-6 w-6 mb-2" />
              Moderation
            </Button>
            <Button variant="outline" size="lg" className="h-20 flex-col">
              <Search className="h-6 w-6 mb-2" />
              Search Videos
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};

// Upload Components
const FileUpload = ({ onFileSelect, acceptedTypes = ".mp4,.mov,.avi,.mkv" }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleChange = useCallback((e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <div className="w-full">
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept={acceptedTypes}
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="text-center">
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <div className="mt-4">
            <p className="text-lg font-medium text-gray-900">
              {selectedFile ? selectedFile.name : 'Drop your video here'}
            </p>
            <p className="text-sm text-gray-500">
              or click to browse files
            </p>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Supports MP4, MOV, AVI, MKV (max 500MB)
          </p>
        </div>
      </div>
      
      {selectedFile && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Video className="h-5 w-5 text-blue-600 mr-2" />
              <div>
                <p className="text-sm font-medium text-blue-900">{selectedFile.name}</p>
                <p className="text-xs text-blue-600">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setSelectedFile(null);
                onFileSelect(null);
              }}
            >
              Remove
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

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

// Analytics Components
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

// Content Moderation Components
// Content Moderation Components
const ModerationAlert = ({ severity, title, description, videoId, timestamp }) => {
  const severityConfig = {
    high: { color: 'danger', icon: AlertTriangle, bgColor: 'bg-red-50 border-red-200' },
    medium: { color: 'warning', icon: Eye, bgColor: 'bg-yellow-50 border-yellow-200' },
    low: { color: 'info', icon: CheckCircle, bgColor: 'bg-blue-50 border-blue-200' }
  };

  const config = severityConfig[severity] || severityConfig.low;
  const Icon = config.icon;

  return (
    <div className={`p-4 rounded-lg border ${config.bgColor}`}>
      <div className="flex items-start">
        <Icon className={`h-5 w-5 mt-0.5 mr-3 ${
          severity === 'high' ? 'text-red-600' : 
          severity === 'medium' ? 'text-yellow-600' : 'text-blue-600'
        }`} />
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900">{title}</h4>
            <Badge variant={config.color} size="sm">{severity.toUpperCase()}</Badge>
          </div>
          <p className="text-sm text-gray-600 mt-1">{description}</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-gray-500">Video ID: {videoId}</span>
            <span className="text-xs text-gray-500">{timestamp}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const ContentFilter = ({ filters, onFilterChange }) => {
  return (
    <div className="flex flex-wrap gap-2">
      <Button
        size="sm"
        variant={filters.severity === 'all' ? 'primary' : 'outline'}
        onClick={() => onFilterChange({ ...filters, severity: 'all' })}
      >
        All Issues
      </Button>
      <Button
        size="sm"
        variant={filters.severity === 'high' ? 'danger' : 'outline'}
        onClick={() => onFilterChange({ ...filters, severity: 'high' })}
      >
        High Risk
      </Button>
      <Button
        size="sm"
        variant={filters.severity === 'medium' ? 'primary' : 'outline'}
        onClick={() => onFilterChange({ ...filters, severity: 'medium' })}
      >
        Medium Risk
      </Button>
      <Button
        size="sm"
        variant={filters.severity === 'low' ? 'secondary' : 'outline'}
        onClick={() => onFilterChange({ ...filters, severity: 'low' })}
      >
        Low Risk
      </Button>
    </div>
  );
};

const Moderation = () => {
  const [filters, setFilters] = useState({ severity: 'all', category: 'all' });
  const [loading, setLoading] = useState(false);

  // Mock moderation alerts - moved inside component to avoid issues
  const mockAlerts = [
    {
      id: 1,
      severity: 'high',
      title: 'Inappropriate Content Detected',
      description: 'Adult content detected at timestamp 2:45-3:12',
      videoId: 'VID_001',
      timestamp: '5 minutes ago',
      category: 'content'
    },
    {
      id: 2,
      severity: 'medium',
      title: 'Profanity in Audio',
      description: 'Strong language detected in speech transcription',
      videoId: 'VID_002', 
      timestamp: '12 minutes ago',
      category: 'language'
    },
    {
      id: 3,
      severity: 'low',
      title: 'Copyright Music',
      description: 'Potential copyrighted music detected',
      videoId: 'VID_003',
      timestamp: '1 hour ago',
      category: 'copyright'
    }
  ];

  const [alerts, setAlerts] = useState(mockAlerts);

  const filteredAlerts = alerts.filter(alert => {
    if (filters.severity !== 'all' && alert.severity !== filters.severity) return false;
    if (filters.category !== 'all' && alert.category !== filters.category) return false;
    return true;
  });

  const handleModerationAction = async (alertId, action) => {
    setLoading(true);
    try {
      // Simulate API call
      setTimeout(() => {
        setAlerts(prev => prev.filter(alert => alert.id !== alertId));
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Moderation action failed:', error);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Content Moderation</h2>
        <Button>
          <Settings className="h-4 w-4 mr-2" />
          Moderation Settings
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard
          title="Pending Reviews"
          value="23"
          icon={Eye}
          color="orange"
        />
        <StatsCard
          title="High Priority"
          value="3"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Resolved Today"
          value="47"
          icon={CheckCircle}
          color="green"
        />
        <StatsCard
          title="Success Rate"
          value="94%"
          icon={Shield}
          color="blue"
        />
      </div>

      <Card 
        title="Content Alerts"
        actions={<ContentFilter filters={filters} onFilterChange={setFilters} />}
      >
        <div className="space-y-4">
          {filteredAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Shield className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No moderation alerts found</p>
            </div>
          ) : (
            filteredAlerts.map((alert) => (
              <div key={alert.id}>
                <ModerationAlert {...alert} />
                <div className="flex justify-end space-x-2 mt-2">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => handleModerationAction(alert.id, 'dismiss')}
                    disabled={loading}
                  >
                    Dismiss
                  </Button>
                  <Button 
                    size="sm" 
                    variant="danger"
                    onClick={() => handleModerationAction(alert.id, 'remove')}
                    disabled={loading}
                  >
                    Take Action
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
};

// Enhanced API Service with new endpoints
const enhancedAPIService = {
  async batchAnalyze(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch(`${this.baseURL}/batch/analyze`, {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  },

  async similaritySearch(videoId, threshold = 0.8) {
    const response = await fetch(`${this.baseURL}/search/similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, threshold })
    });
    
    return response.json();
  },

  async extractHighlights(videoId, maxDuration = 300) {
    const response = await fetch(`${this.baseURL}/extract/highlights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, max_duration: maxDuration })
    });
    
    return response.json();
  }
};

// Main Application Component
const VidIntelPro = () => {
  const [activeView, setActiveView] = useState('dashboard');
  const [platformStats, setPlatformStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const api = useAPI();

  useEffect(() => {
    const loadPlatformStats = async () => {
      try {
        setLoading(true);
        const stats = await api.getPlatformStats();
        console.log('Fetched platform stats:', stats);
        setPlatformStats(stats);
      } catch (error) {
        console.error('Failed to load platform stats:', error);
      } finally {
        setLoading(false);
      }
    };

    loadPlatformStats();
  }, [api]);

  const renderActiveView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard platformStats={platformStats} />;
      case 'upload':
        return <UploadSection />;
      case 'analytics':
        return <Analytics platformStats={platformStats} />;
      case 'moderation':
        return <Moderation />;
      default:
        return <Dashboard platformStats={platformStats} />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="xl" />
          <p className="mt-4 text-gray-600">Loading VidIntel Pro...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeView={activeView} onViewChange={setActiveView} />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveView()}
      </main>
    </div>
  );
};

export default VidIntelPro;