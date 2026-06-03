import { useState, useCallback } from 'react';
import TeacherLogin from './components/TeacherLogin';
import Dashboard from './components/Dashboard';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return !!sessionStorage.getItem('teacher_token');
  });

  const handleLoginSuccess = useCallback(() => {
    setIsAuthenticated(true);
  }, []);

  const handleLogout = useCallback(() => {
    sessionStorage.removeItem('teacher_token');
    setIsAuthenticated(false);
  }, []);

  if (!isAuthenticated) {
    return <TeacherLogin onLoginSuccess={handleLoginSuccess} />;
  }

  return <Dashboard onLogout={handleLogout} />;
}
