/**
 * SessionManager - Handles session creation, data persistence, and session management
 * for video analysis workflows
 */

class SessionManager {
  constructor() {
    this.SESSION_PREFIX = 'analysis_';
    this.ACTIVE_SESSIONS_KEY = 'active_analysis_sessions';
  }

  /**
   * Generate a unique session ID
   * @returns {string} Unique session identifier
   */
  generateSessionId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    return `${timestamp}_${random}`;
  }

  /**
   * Create a new analysis session
   * @param {File} videoFile - The video file to analyze
   * @param {Object} options - Analysis options
   * @returns {string} Session ID
   */
  createSession(videoFile, options = {}) {
    const sessionId = this.generateSessionId();
    
    const sessionData = {
      sessionId,
      videoFile: {
        name: videoFile.name,
        size: videoFile.size,
        type: videoFile.type,
        lastModified: videoFile.lastModified
      },
      options,
      status: 'created',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    // Store session data
    this.saveSessionData(sessionId, sessionData);
    
    // Add to active sessions list
    this.addToActiveSessions(sessionId);
    
    return sessionId;
  }

  /**
   * Save analysis data to session storage
   * @param {string} sessionId - Session identifier
   * @param {Object} data - Data to save
   */
  saveSessionData(sessionId, data) {
    try {
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      const dataWithTimestamp = {
        ...data,
        updatedAt: new Date().toISOString()
      };
      sessionStorage.setItem(sessionKey, JSON.stringify(dataWithTimestamp));
    } catch (error) {
      console.error('Failed to save session data:', error);
      throw new Error('Failed to save session data');
    }
  }

  /**
   * Get analysis data from session storage
   * @param {string} sessionId - Session identifier
   * @returns {Object|null} Session data or null if not found
   */
  getSessionData(sessionId) {
    try {
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      const data = sessionStorage.getItem(sessionKey);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Failed to get session data:', error);
      return null;
    }
  }

  /**
   * Update existing session data
   * @param {string} sessionId - Session identifier
   * @param {Object} updates - Data updates to apply
   */
  updateSessionData(sessionId, updates) {
    const existingData = this.getSessionData(sessionId);
    if (!existingData) {
      throw new Error(`Session ${sessionId} not found`);
    }

    const updatedData = {
      ...existingData,
      ...updates,
      updatedAt: new Date().toISOString()
    };

    this.saveSessionData(sessionId, updatedData);
  }

  /**
   * Delete session data
   * @param {string} sessionId - Session identifier
   */
  deleteSession(sessionId) {
    try {
      const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
      sessionStorage.removeItem(sessionKey);
      this.removeFromActiveSessions(sessionId);
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  }

  /**
   * Get all active sessions
   * @returns {Array} Array of active session IDs
   */
  getActiveSessions() {
    try {
      const sessions = sessionStorage.getItem(this.ACTIVE_SESSIONS_KEY);
      return sessions ? JSON.parse(sessions) : [];
    } catch (error) {
      console.error('Failed to get active sessions:', error);
      return [];
    }
  }

  /**
   * Add session to active sessions list
   * @param {string} sessionId - Session identifier
   */
  addToActiveSessions(sessionId) {
    try {
      const activeSessions = this.getActiveSessions();
      if (!activeSessions.includes(sessionId)) {
        activeSessions.push(sessionId);
        sessionStorage.setItem(this.ACTIVE_SESSIONS_KEY, JSON.stringify(activeSessions));
      }
    } catch (error) {
      console.error('Failed to add to active sessions:', error);
    }
  }

  /**
   * Remove session from active sessions list
   * @param {string} sessionId - Session identifier
   */
  removeFromActiveSessions(sessionId) {
    try {
      const activeSessions = this.getActiveSessions();
      const filteredSessions = activeSessions.filter(id => id !== sessionId);
      sessionStorage.setItem(this.ACTIVE_SESSIONS_KEY, JSON.stringify(filteredSessions));
    } catch (error) {
      console.error('Failed to remove from active sessions:', error);
    }
  }

  /**
   * Check if session exists
   * @param {string} sessionId - Session identifier
   * @returns {boolean} True if session exists
   */
  sessionExists(sessionId) {
    return this.getSessionData(sessionId) !== null;
  }

  /**
   * Clear all session data (useful for cleanup)
   */
  clearAllSessions() {
    try {
      const activeSessions = this.getActiveSessions();
      activeSessions.forEach(sessionId => {
        const sessionKey = `${this.SESSION_PREFIX}${sessionId}`;
        sessionStorage.removeItem(sessionKey);
      });
      sessionStorage.removeItem(this.ACTIVE_SESSIONS_KEY);
    } catch (error) {
      console.error('Failed to clear all sessions:', error);
    }
  }

  /**
   * Get session status
   * @param {string} sessionId - Session identifier
   * @returns {string|null} Session status or null if not found
   */
  getSessionStatus(sessionId) {
    const data = this.getSessionData(sessionId);
    return data ? data.status : null;
  }

  /**
   * Update session status
   * @param {string} sessionId - Session identifier
   * @param {string} status - New status
   */
  updateSessionStatus(sessionId, status) {
    this.updateSessionData(sessionId, { status });
  }

  /**
   * Get session creation time
   * @param {string} sessionId - Session identifier
   * @returns {string|null} ISO timestamp or null if not found
   */
  getSessionCreatedAt(sessionId) {
    const data = this.getSessionData(sessionId);
    return data ? data.createdAt : null;
  }

  /**
   * Get session last update time
   * @param {string} sessionId - Session identifier
   * @returns {string|null} ISO timestamp or null if not found
   */
  getSessionUpdatedAt(sessionId) {
    const data = this.getSessionData(sessionId);
    return data ? data.updatedAt : null;
  }

  /**
   * Clean up old sessions (older than specified hours)
   * @param {number} hoursOld - Hours threshold for cleanup
   */
  cleanupOldSessions(hoursOld = 24) {
    try {
      const activeSessions = this.getActiveSessions();
      const cutoffTime = new Date(Date.now() - (hoursOld * 60 * 60 * 1000));
      
      activeSessions.forEach(sessionId => {
        const data = this.getSessionData(sessionId);
        if (data && new Date(data.updatedAt) < cutoffTime) {
          this.deleteSession(sessionId);
        }
      });
    } catch (error) {
      console.error('Failed to cleanup old sessions:', error);
    }
  }
}

// Create and export a singleton instance
const sessionManager = new SessionManager();

export default sessionManager;