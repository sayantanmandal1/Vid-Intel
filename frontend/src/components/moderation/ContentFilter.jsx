import React from 'react';
import { Button } from '../common';

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

export default ContentFilter;