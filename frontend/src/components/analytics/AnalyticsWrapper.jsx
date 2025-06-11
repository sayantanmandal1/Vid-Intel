// src/components/dashboard/AnalyticsWrapper.jsx

import React from 'react';
import { useParams } from 'react-router-dom';
import Analytics from './Analytics';

const AnalyticsWrapper = () => {
  const { videoId } = useParams();
  return <Analytics videoId={videoId} />;
};

export default AnalyticsWrapper;
