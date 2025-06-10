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

  async batchAnalyze(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch(`${this.baseURL}/batch/analyze`, {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  }

  async similaritySearch(videoId, threshold = 0.8) {
    const response = await fetch(`${this.baseURL}/search/similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, threshold })
    });
    
    return response.json();
  }

  async extractHighlights(videoId, maxDuration = 300) {
    const response = await fetch(`${this.baseURL}/extract/highlights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, max_duration: maxDuration })
    });
    
    return response.json();
  }
}

export default APIService;