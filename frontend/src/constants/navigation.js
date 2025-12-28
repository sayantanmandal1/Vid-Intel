import { 
  Home, 
  Upload, 
  BarChart3, 
  Shield, 
  Settings, 
  FileText,
  Users,
  Bell
} from 'lucide-react';

export const NAVIGATION_ITEMS = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: Home,
    path: '/',
    active: true
  },
  {
    id: 'upload',
    label: 'Upload & Analyze',
    icon: Upload,
    path: '/upload'
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: BarChart3,
    path: '/analytics'
  },
  {
    id: 'moderation',
    label: 'Content Moderation',
    icon: Shield,
    path: '/moderation'
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: FileText,
    path: '/reports'
  },
  {
    id: 'users',
    label: 'User Management',
    icon: Users,
    path: '/users'
  },
  {
    id: 'notifications',
    label: 'Notifications',
    icon: Bell,
    path: '/notifications'
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    path: '/settings'
  }
];

export const QUICK_ACTIONS = [
  {
    id: 'upload-video',
    label: 'Upload Video',
    icon: Upload,
    color: 'blue',
    path: '/upload'
  },
  {
    id: 'view-reports',
    label: 'View Reports',
    icon: FileText,
    color: 'green',
    path: '/reports'
  },
  {
    id: 'moderation-queue',
    label: 'Moderation Queue',
    icon: Shield,
    color: 'orange',
    path: '/moderation'
  }
];