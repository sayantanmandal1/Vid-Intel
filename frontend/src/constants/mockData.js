export const MOCK_STATS = {
  totalVideos: 1247,
  totalAnalyzed: 1189,
  processingTime: '2.3m',
  successRate: '95.3%'
};

export const MOCK_RECENT_ACTIVITY = [
  {
    id: 1,
    type: 'upload',
    description: 'New video uploaded: marketing_demo.mp4',
    timestamp: '2 minutes ago',
    status: 'completed'
  },
  {
    id: 2,
    type: 'analysis',
    description: 'Sentiment analysis completed for product_review.mp4',
    timestamp: '5 minutes ago',
    status: 'completed'
  },
  {
    id: 3,
    type: 'moderation',
    description: 'Content flagged for manual review',
    timestamp: '12 minutes ago',
    status: 'pending'
  },
  {
    id: 4,
    type: 'export',
    description: 'Analytics report exported',
    timestamp: '1 hour ago',
    status: 'completed'
  }
];

export const MOCK_ANALYTICS_DATA = {
  uploadTrends: [
    { date: '2024-01-01', uploads: 12 },
    { date: '2024-01-02', uploads: 19 },
    { date: '2024-01-03', uploads: 15 },
    { date: '2024-01-04', uploads: 22 },
    { date: '2024-01-05', uploads: 18 },
    { date: '2024-01-06', uploads: 25 },
    { date: '2024-01-07', uploads: 31 }
  ],
  analysisTypes: [
    { name: 'Transcription', value: 45 },
    { name: 'Sentiment', value: 30 },
    { name: 'Face Detection', value: 15 },
    { name: 'Scene Detection', value: 10 }
  ],
  contentCategories: [
    { category: 'Marketing', count: 45 },
    { category: 'Training', count: 32 },
    { category: 'Product Demo', count: 28 },
    { category: 'Educational', count: 21 },
    { category: 'Entertainment', count: 15 }
  ]
};

export const MOCK_MODERATION_ALERTS = [
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
  },
  {
    id: 4,
    severity: 'medium',
    title: 'Violence Detection',
    description: 'Aggressive behavior detected in video frames',
    videoId: 'VID_004',
    timestamp: '2 hours ago',
    category: 'violence'
  }
];

export const MOCK_PLATFORM_STATS = {
  totalUsers: 2547,
  activeUsers: 1829,
  totalStorage: '2.4 TB',
  processingCapacity: '85%'
};