'use client';

import { useEffect, useState } from 'react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

let toastId = 0;
const listeners: Set<(toast: Toast) => void> = new Set();

export function showToast(message: string, type: ToastType = 'info') {
  const toast: Toast = { id: ++toastId, message, type };
  listeners.forEach(listener => listener(toast));
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const listener = (toast: Toast) => {
      setToasts(prev => [...prev, toast]);
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== toast.id));
      }, 4000);
    };
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-24 right-4 z-50 flex flex-col gap-2">
      {toasts.map(toast => (
        <div
          key={toast.id}
          className={`px-4 py-3 rounded-lg shadow-lg text-sm font-medium animate-slide-in ${
            toast.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' :
            toast.type === 'error' ? 'bg-red-50 text-red-800 border border-red-200' :
            toast.type === 'warning' ? 'bg-yellow-50 text-yellow-800 border border-yellow-200' :
            'bg-blue-50 text-blue-800 border border-blue-200'
          }`}
        >
          <div className="flex items-center gap-2">
            {toast.type === 'success' && <span className="text-green-500">✓</span>}
            {toast.type === 'error' && <span className="text-red-500">✕</span>}
            {toast.type === 'warning' && <span className="text-yellow-500">⚠</span>}
            {toast.type === 'info' && <span className="text-blue-500">ℹ</span>}
            <span>{toast.message}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
