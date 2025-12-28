import { useState } from 'react';
import APIService from '../services/api';

export const useAPI = () => {
  const [api] = useState(() => new APIService());
  return api;
};