import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { 
  Upload, Play, Pause, SkipForward, SkipBack, Volume2, VolumeX, 
  Download, Share2, Settings, Menu, X, ChevronDown, ChevronUp,
  BarChart3, TrendingUp, Users, Clock, FileText, Eye, Brain,
  AlertTriangle, CheckCircle, XCircle, Info, Zap, Search,
  Filter, SortAsc, SortDesc, RefreshCw, Bell, User, LogOut,
  Globe, Mic, Camera, FileVideo, Heart, Star, MessageSquare,
  PieChart as PieChartIcon, LineChart as LineChartIcon, // 👈 Alias icons to avoid conflict
  Activity, Target, Layers, Database
} from 'lucide-react';

import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell, ResponsiveContainer, XAxis,
  YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const VidIntelPro = () => {
  // Core state management
  const [currentView, setCurrentView] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState({});
  const [videos, setVideos] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterOptions, setFilterOptions] = useState({
    status: 'all',
    type: 'all',
    dateRange: '7d'
  });

  // Video player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef(null);

  // Mock data for demonstration
  const mockVideos = useMemo(() => [
    {
      id: '1',
      filename: 'corporate_presentation.mp4',
      duration: 1800,
      fileSize: 245000000,
      uploadTime: '2024-06-10T14:30:00',
      status: 'completed',
      thumbnail: '/api/placeholder/400/225',
      analysis: {
        sentiment: { positive: 0.75, negative: 0.15, neutral: 0.10 },
        topics: ['business strategy', 'quarterly results', 'market analysis'],
        toxicity: 0.05,
        engagement: 0.82,
        faces: 3,
        scenes: 12
      }
    },
    {
      id: '2',
      filename: 'marketing_webinar.mp4',
      duration: 3600,
      fileSize: 450000000,
      uploadTime: '2024-06-10T10:15:00',
      status: 'processing',
      thumbnail: '/api/placeholder/400/225',
      analysis: {
        sentiment: { positive: 0.68, negative: 0.20, neutral: 0.12 },
        topics: ['digital marketing', 'social media', 'ROI optimization'],
        toxicity: 0.08,
        engagement: 0.76,
        faces: 2,
        scenes: 18
      }
    },
    {
      id: '3',
      filename: 'team_meeting.mp4',
      duration: 2700,
      fileSize: 320000000,
      uploadTime: '2024-06-09T16:45:00',
      status: 'completed',
      thumbnail: '/api/placeholder/400/225',
      analysis: {
        sentiment: { positive: 0.60, negative: 0.25, neutral: 0.15 },
        topics: ['project updates', 'deadlines', 'resource allocation'],
        toxicity: 0.12,
        engagement: 0.65,
        faces: 6,
        scenes: 8
      }
    }
  ], []);

  const platformStats = useMemo(() => ({
    totalVideos: 1247,
    totalHours: 3521,
    avgProcessingTime: '4.2min',
    accuracyRate: '94.7%',
    activeUsers: 89,
    todayUploads: 23
  }), []);

  const engagementData = useMemo(() => [
    { time: '00:00', engagement: 20 },
    { time: '05:00', engagement: 45 },
    { time: '10:00', engagement: 65 },
    { time: '15:00', engagement: 40 },
    { time: '20:00', engagement: 80 },
    { time: '25:00', engagement: 70 },
    { time: '30:00', engagement: 55 }
  ], []);

  const sentimentData = useMemo(() => [
    { name: 'Positive', value: 65, color: '#10B981' },
    { name: 'Neutral', value: 25, color: '#6B7280' },
    { name: 'Negative', value: 10, color: '#EF4444' }
  ], []);

  // Effects
  useEffect(() => {
    setVideos(mockVideos);
  }, [mockVideos]);

  useEffect(() => {
    // Simulate real-time notifications
    const interval = setInterval(() => {
      if (Math.random() > 0.8) {
        const newNotification = {
          id: Date.now(),
          type: 'success',
          message: `Video analysis completed successfully`,
          timestamp: new Date()
        };
        setNotifications(prev => [newNotification, ...prev.slice(0, 4)]);
      }
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // Handlers
  const handleVideoUpload = useCallback(async (files) => {
    const file = files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          
          // Add new video to the list
          const newVideo = {
            id: Date.now().toString(),
            filename: file.name,
            duration: 0,
            fileSize: file.size,
            uploadTime: new Date().toISOString(),
            status: 'processing',
            thumbnail: '/api/placeholder/400/225',
            analysis: null
          };
          setVideos(prev => [newVideo, ...prev]);
          
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
  }, []);

  const handleVideoSelect = useCallback((video) => {
    setSelectedVideo(video);
    setCurrentView('analysis');
  }, []);

  const formatDuration = useCallback((seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const formatFileSize = useCallback((bytes) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }, []);

  // Filtered videos based on search and filters
  const filteredVideos = useMemo(() => {
    return videos.filter(video => {
      const matchesSearch = video.filename.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = filterOptions.status === 'all' || video.status === filterOptions.status;
      return matchesSearch && matchesStatus;
    });
  }, [videos, searchQuery, filterOptions]);

  // Navigation component
  const Navigation = () => (
    <nav className={`bg-white border-r border-gray-200 transition-all duration-300 ${
      sidebarCollapsed ? 'w-16' : 'w-64'
    } flex flex-col h-full`}>
      {/* Logo and toggle */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        {!sidebarCollapsed && (
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              VidIntel Pro
            </span>
          </div>
        )}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500"
        >
          <Menu className="w-5 h-5" />
        </button>
      </div>

      {/* Navigation items */}
      <div className="flex-1 p-4 space-y-2">
        {[
          { id: 'dashboard', icon: BarChart3, label: 'Dashboard' },
          { id: 'videos', icon: FileVideo, label: 'Videos' },
          { id: 'analytics', icon: TrendingUp, label: 'Analytics' },
          { id: 'batch', icon: Layers, label: 'Batch Processing' },
          { id: 'moderation', icon: AlertTriangle, label: 'Moderation' },
          { id: 'settings', icon: Settings, label: 'Settings' }
        ].map(item => (
          <button
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
              currentView === item.id 
                ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <item.icon className="w-5 h-5" />
            {!sidebarCollapsed && <span className="font-medium">{item.label}</span>}
          </button>
        ))}
      </div>

      {/* User section */}
      <div className="border-t border-gray-200 p-4">
        <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'space-x-3'}`}>
          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          {!sidebarCollapsed && (
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">John Doe</p>
              <p className="text-xs text-gray-500">Admin</p>
            </div>
          )}
        </div>
      </div>
    </nav>
  );

  // Header component
  const Header = () => (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 capitalize">{currentView}</h1>
          <p className="text-sm text-gray-500">
            {currentView === 'dashboard' && 'Overview of your video intelligence platform'}
            {currentView === 'videos' && 'Manage and analyze your video content'}
            {currentView === 'analytics' && 'Deep insights and performance metrics'}
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Search */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-64"
            />
          </div>

          {/* Notifications */}
          <div className="relative">
            <button className="p-2 text-gray-400 hover:text-gray-600 relative">
              <Bell className="w-5 h-5" />
              {notifications.length > 0 && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
              )}
            </button>
          </div>

          {/* Upload button */}
          <label className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 cursor-pointer transition-all">
            <Upload className="w-4 h-4" />
            <span>Upload Video</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => handleVideoUpload(e.target.files)}
              className="hidden"
            />
          </label>
        </div>
      </div>
    </header>
  );

  // Dashboard view
  const DashboardView = () => (
    <div className="p-6 space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { label: 'Total Videos', value: platformStats.totalVideos, icon: FileVideo, color: 'blue', change: '+12%' },
          { label: 'Processing Hours', value: `${platformStats.totalHours}h`, icon: Clock, color: 'green', change: '+8%' },
          { label: 'Active Users', value: platformStats.activeUsers, icon: Users, color: 'purple', change: '+5%' },
          { label: 'Accuracy Rate', value: platformStats.accuracyRate, icon: Target, color: 'pink', change: '+2%' }
        ].map((stat, index) => (
          <div key={index} className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                <p className={`text-sm font-medium mt-1 text-${stat.color}-600`}>
                  {stat.change} from last month
                </p>
              </div>
              <div className={`p-3 bg-${stat.color}-100 rounded-lg`}>
                <stat.icon className={`w-6 h-6 text-${stat.color}-600`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Engagement over time */}
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Engagement Timeline</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={engagementData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="engagement" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Sentiment distribution */}
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="value"
              >
                {sentimentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent videos */}
      <div className="bg-white rounded-xl border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Videos</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {videos.slice(0, 5).map(video => (
            <div key={video.id} className="p-6 hover:bg-gray-50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="w-16 h-12 bg-gray-200 rounded-lg overflow-hidden">
                    <img src={video.thumbnail} alt={video.filename} className="w-full h-full object-cover" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">{video.filename}</h4>
                    <p className="text-sm text-gray-500">
                      {formatDuration(video.duration)} • {formatFileSize(video.fileSize)}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    video.status === 'completed' ? 'bg-green-100 text-green-800' :
                    video.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {video.status}
                  </span>
                  <button
                    onClick={() => handleVideoSelect(video)}
                    className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                  >
                    View Analysis
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // Videos view
  const VideosView = () => (
    <div className="p-6">
      {/* Filters */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <select
              value={filterOptions.status}
              onChange={(e) => setFilterOptions(prev => ({ ...prev, status: e.target.value }))}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
            </select>
            
            <select
              value={filterOptions.dateRange}
              onChange={(e) => setFilterOptions(prev => ({ ...prev, dateRange: e.target.value }))}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <button className="p-2 text-gray-400 hover:text-gray-600">
              <SortAsc className="w-4 h-4" />
            </button>
            <button className="p-2 text-gray-400 hover:text-gray-600">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Videos grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredVideos.map(video => (
          <div key={video.id} className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow">
            <div className="aspect-video bg-gray-200 relative overflow-hidden">
              <img src={video.thumbnail} alt={video.filename} className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-black bg-opacity-40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
                <button
                  onClick={() => handleVideoSelect(video)}
                  className="bg-white bg-opacity-90 rounded-full p-3 hover:bg-opacity-100 transition-all"
                >
                  <Play className="w-6 h-6 text-gray-800" />
                </button>
              </div>
              <div className="absolute top-2 right-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  video.status === 'completed' ? 'bg-green-500 text-white' :
                  video.status === 'processing' ? 'bg-yellow-500 text-white' :
                  'bg-red-500 text-white'
                }`}>
                  {video.status}
                </span>
              </div>
              <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                {formatDuration(video.duration)}
              </div>
            </div>
            
            <div className="p-4">
              <h3 className="font-medium text-gray-900 truncate">{video.filename}</h3>
              <p className="text-sm text-gray-500 mt-1">
                {formatFileSize(video.fileSize)} • {new Date(video.uploadTime).toLocaleDateString()}
              </p>
              
              {video.analysis && (
                <div className="mt-3 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">Engagement</span>
                    <span className="font-medium">{Math.round(video.analysis.engagement * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div 
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${video.analysis.engagement * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
              
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  {video.analysis && (
                    <>
                      <span className="flex items-center">
                        <Eye className="w-3 h-3 mr-1" />
                        {video.analysis.faces}
                      </span>
                      <span className="flex items-center">
                        <Camera className="w-3 h-3 mr-1" />
                        {video.analysis.scenes}
                      </span>
                    </>
                  )}
                </div>
                <button
                  onClick={() => handleVideoSelect(video)}
                  className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                >
                  Analyze
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {filteredVideos.length === 0 && (
        <div className="text-center py-12">
          <FileVideo className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No videos found</h3>
          <p className="text-gray-500">Upload your first video to get started with AI-powered analysis.</p>
        </div>
      )}
    </div>
  );

  // Analysis view
  const AnalysisView = () => {
    if (!selectedVideo) {
      return (
        <div className="p-6 text-center">
          <FileVideo className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Select a video to analyze</h3>
          <p className="text-gray-500">Choose a video from your library to view detailed analysis results.</p>
        </div>
      );
    }

    return (
      <div className="p-6 space-y-6">
        {/* Video player section */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="aspect-video bg-black relative">
            <video
              ref={videoRef}
              className="w-full h-full"
              poster={selectedVideo.thumbnail}
              onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
              onLoadedMetadata={(e) => setDuration(e.target.duration)}
            >
              <source src={`/api/videos/${selectedVideo.id}/stream`} type="video/mp4" />
            </video>
            
            {/* Video controls */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
              <div className="flex items-center space-x-4 text-white">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-2 hover:bg-white hover:bg-opacity-20 rounded-full transition-colors"
                >
                  {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                </button>
                
                <div className="flex items-center space-x-2 flex-1">
                  <span className="text-sm">{formatDuration(currentTime)}</span>
                  <div className="flex-1 bg-white bg-opacity-30 rounded-full h-1">
                    <div 
                      className="bg-white h-1 rounded-full transition-all"
                      style={{ width: `${(currentTime / duration) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm">{formatDuration(duration)}</span>
                </div>
                
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className="p-2 hover:bg-white hover:bg-opacity-20 rounded-full transition-colors"
                >
                  {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                </button>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-900">{selectedVideo.filename}</h2>
            <p className="text-gray-500 mt-1">
              {formatDuration(selectedVideo.duration)} • {formatFileSize(selectedVideo.fileSize)} • 
              Uploaded {new Date(selectedVideo.uploadTime).toLocaleDateString()}
            </p>
          </div>
        </div>

        {selectedVideo.analysis && (
          <>
            {/* Analysis overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Sentiment Score</p>
                    <p className="text-2xl font-bold text-green-600 mt-1">
                      {Math.round(selectedVideo.analysis.sentiment.positive * 100)}%
                    </p>
                  </div>
                  <Heart className="w-8 h-8 text-green-500" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Engagement</p>
                    <p className="text-2xl font-bold text-blue-600 mt-1">
                      {Math.round(selectedVideo.analysis.engagement * 100)}%
                    </p>
                  </div>
                  <Activity className="w-8 h-8 text-blue-500" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Toxicity Level</p>
                    <p className="text-2xl font-bold text-red-600 mt-1">
                      {Math.round(selectedVideo.analysis.toxicity * 100)}%
                    </p>
                  </div>
                  <AlertTriangle className="w-8 h-8 text-red-500" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Faces Detected</p>
                    <p className="text-2xl font-bold text-purple-600 mt-1">
                      {selectedVideo.analysis.faces}
                    </p>
                  </div>
                  <Eye className="w-8 h-8 text-purple-500" />
                </div>
              </div>
            </div>

            {/* Detailed analysis sections */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment breakdown */}
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Breakdown</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">Positive</span>
                    <span className="text-sm font-bold text-green-600">
                      {Math.round(selectedVideo.analysis.sentiment.positive * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all"
                      style={{ width: `${selectedVideo.analysis.sentiment.positive * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">Neutral</span>
                    <span className="text-sm font-bold text-gray-600">
                      {Math.round(selectedVideo.analysis.sentiment.neutral * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gray-500 h-2 rounded-full transition-all"
                      style={{ width: `${selectedVideo.analysis.sentiment.neutral * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">Negative</span>
                    <span className="text-sm font-bold text-red-600">
                      {Math.round(selectedVideo.analysis.sentiment.negative * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full transition-all"
                      style={{ width: `${selectedVideo.analysis.sentiment.negative * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Key topics */}
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Topics</h3>
                <div className="space-y-3">
                  {selectedVideo.analysis.topics.map((topic, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm font-medium text-gray-900 capitalize">{topic}</span>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                        Topic {index + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Engagement timeline */}
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Engagement Timeline</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={engagementData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="engagement" stroke="#3B82F6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    );
  };

  // Analytics view
  const AnalyticsView = () => (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Platform overview */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Volume</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { month: 'Jan', videos: 120 },
                { month: 'Feb', videos: 135 },
                { month: 'Mar', videos: 148 },
                { month: 'Apr', videos: 162 },
                { month: 'May', videos: 175 },
                { month: 'Jun', videos: 189 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="videos" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quick stats */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Avg Processing Time</span>
                <span className="font-semibold text-gray-900">{platformStats.avgProcessingTime}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="font-semibold text-green-600">{platformStats.accuracyRate}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Today's Uploads</span>
                <span className="font-semibold text-blue-600">{platformStats.todayUploads}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Batch processing view
  const BatchView = () => (
    <div className="p-6 space-y-6">
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Batch Processing</h3>
        <div className="text-center py-12">
          <Layers className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h4 className="text-lg font-medium text-gray-900 mb-2">Batch Processing</h4>
          <p className="text-gray-500 mb-6">Process multiple videos simultaneously for efficiency.</p>
          <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Start Batch Process
          </button>
        </div>
      </div>
    </div>
  );

  // Moderation view
  const ModerationView = () => (
    <div className="p-6 space-y-6">
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Content Moderation</h3>
        <div className="text-center py-12">
          <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h4 className="text-lg font-medium text-gray-900 mb-2">Content Moderation</h4>
          <p className="text-gray-500 mb-6">Monitor and moderate content for safety and compliance.</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 border border-gray-200 rounded-lg">
              <h5 className="font-medium text-gray-900">Flagged Content</h5>
              <p className="text-2xl font-bold text-red-600 mt-2">3</p>
            </div>
            <div className="p-4 border border-gray-200 rounded-lg">
              <h5 className="font-medium text-gray-900">Under Review</h5>
              <p className="text-2xl font-bold text-yellow-600 mt-2">7</p>
            </div>
            <div className="p-4 border border-gray-200 rounded-lg">
              <h5 className="font-medium text-gray-900">Approved</h5>
              <p className="text-2xl font-bold text-green-600 mt-2">142</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Settings view
  const SettingsView = () => (
    <div className="p-6 space-y-6">
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">Platform Settings</h3>
        <div className="space-y-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Analysis Settings</h4>
            <div className="space-y-3">
              <label className="flex items-center space-x-3">
                <input type="checkbox" className="rounded border-gray-300" defaultChecked />
                <span className="text-sm text-gray-700">Enable sentiment analysis</span>
              </label>
              <label className="flex items-center space-x-3">
                <input type="checkbox" className="rounded border-gray-300" defaultChecked />
                <span className="text-sm text-gray-700">Enable face detection</span>
              </label>
              <label className="flex items-center space-x-3">
                <input type="checkbox" className="rounded border-gray-300" defaultChecked />
                <span className="text-sm text-gray-700">Enable content moderation</span>
              </label>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Processing Settings</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Processing Quality
                </label>
                <select className="w-full border border-gray-300 rounded-lg px-3 py-2">
                  <option>High Quality (Slower)</option>
                  <option>Standard Quality</option>
                  <option>Fast Processing</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Upload progress modal
  const UploadModal = () => {
    if (!isUploading) return null;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Uploading Video</h3>
          <div className="space-y-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 text-center">
              {Math.round(uploadProgress)}% completed
            </p>
          </div>
        </div>
      </div>
    );
  };

  // Render current view
  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return <DashboardView />;
      case 'videos':
        return <VideosView />;
      case 'analysis':
        return <AnalysisView />;
      case 'analytics':
        return <AnalyticsView />;
      case 'batch':
        return <BatchView />;
      case 'moderation':
        return <ModerationView />;
      case 'settings':
        return <SettingsView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Navigation />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto">
          {renderCurrentView()}
        </main>
      </div>
      <UploadModal />
    </div>
  );
};

export default VidIntelPro;