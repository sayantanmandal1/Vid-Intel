class APIService {
  constructor(baseURL = 'http://127.0.0.1:8000') {
    this.baseURL = baseURL;
  }

  async comprehensiveAnalysis(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/analyze/comprehensive`, {
      method: 'POST',
      body: formData
    });

    return response.json();
  }

  async quickAnalysis(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/analyze/quick`, {
      method: 'POST',
      body: formData
    });

    return response.json();
  }

  async contentModeration(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/analyze/content_moderation`, {
      method: 'POST',
      body: formData
    });

    return response.json();
  }

  async engagementAnalysis(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/analyze/engagement_metrics`, {
      method: 'POST',
      body: formData
    });

    return response.json();
  }

  async batchAnalyze(files, analysisType = 'quick') {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('analysis_type', analysisType);

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

  async generateChapters(videoId) {
    const response = await fetch(`${this.baseURL}/generate/chapters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId })
    });

    return response.json();
  }

  async getAnalyticsDashboard(videoId) {
    const response = await fetch(`${this.baseURL}/analytics/dashboard/${videoId}`, {
      method: 'GET'
    });

    return response.json();
  }

  async getPlatformStats() {
    const response = await fetch(`${this.baseURL}/stats/platform`, {
      method: 'GET'
    });

    return response.json();
  }

  async analyzeVideo(videoFile, sessionId) {
    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('session_id', sessionId);

    const response = await fetch(`${this.baseURL}/analyze/video`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export default APIService;
