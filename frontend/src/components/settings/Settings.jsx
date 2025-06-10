import React, { useState } from 'react';
import { Card, Button, Badge } from '../common';
import { ToggleLeft, ToggleRight, Save, RefreshCw } from 'lucide-react';

const Settings = () => {
  const [settings, setSettings] = useState({
    enableNotifications: true,
    autoSaveTranscripts: false,
    darkMode: false,
    analyticsTracking: true
  });

  const toggleSetting = (key) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = () => {
    console.log('Saved settings:', settings);
    // Save logic here (e.g., API call)
  };

  const SettingRow = ({ label, description, settingKey }) => (
    <div className="flex items-start justify-between py-4 border-b">
      <div>
        <p className="text-sm font-medium text-gray-900">{label}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <button
        onClick={() => toggleSetting(settingKey)}
        className="focus:outline-none"
        aria-label={`Toggle ${label}`}
      >
        {settings[settingKey] ? (
          <ToggleRight className="text-green-600 w-6 h-6" />
        ) : (
          <ToggleLeft className="text-gray-400 w-6 h-6" />
        )}
      </button>
    </div>
  );

  return (
    <div className="max-w-3xl mx-auto space-y-6 py-10">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">Settings</h1>

      <Card title="Preferences">
        <SettingRow
          label="Enable Notifications"
          description="Receive alerts and updates about your analysis progress."
          settingKey="enableNotifications"
        />
        <SettingRow
          label="Auto-Save Transcripts"
          description="Automatically save transcripts to your account."
          settingKey="autoSaveTranscripts"
        />
        <SettingRow
          label="Enable Dark Mode"
          description="Switch to a dark theme for better night-time viewing."
          settingKey="darkMode"
        />
        <SettingRow
          label="Analytics Tracking"
          description="Allow anonymous usage analytics to improve features."
          settingKey="analyticsTracking"
        />

        <div className="pt-4 flex justify-end space-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button size="sm" onClick={handleSave}>
            <Save className="w-4 h-4 mr-2" />
            Save Changes
          </Button>
        </div>
      </Card>

      <Card title="Current Status">
        <div className="flex flex-wrap gap-2">
          <Badge variant="info">Version 1.2.4</Badge>
          <Badge variant="success">
            {settings.darkMode ? 'Dark Mode Enabled' : 'Light Mode'}
          </Badge>
          <Badge variant={settings.analyticsTracking ? 'purple' : 'warning'}>
            Analytics: {settings.analyticsTracking ? 'On' : 'Off'}
          </Badge>
        </div>
      </Card>
    </div>
  );
};

export default Settings;
