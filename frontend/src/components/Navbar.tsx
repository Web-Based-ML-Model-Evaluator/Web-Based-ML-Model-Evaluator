import React, { useEffect, useState } from 'react';
import { Terminal, Cpu, Play, Search, Activity, CheckSquare } from 'lucide-react';
import { BackendApi } from '../services/backendApi';

interface NavbarProps {
  activeTab: 'train' | 'evaluate';
  onTabChange: (tab: 'train' | 'evaluate') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange }) => {
  const [backendStatus, setBackendStatus] = useState<{ online: boolean; checking: boolean }>({
    online: false,
    checking: true,
  });

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const res = await BackendApi.checkHealth();
      if (mounted) {
        setBackendStatus({ online: res.online, checking: false });
      }
    };
    check();
    const interval = setInterval(check, 6000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="border-b-2 border-[#101010] bg-[#FAF5ED] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="bg-[#101010] text-[#FF5500] font-mono font-black text-base px-2.5 py-1 border-2 border-[#101010] shadow-[2px_2px_0px_#FF5500]">
            ML//SYS
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight text-[#101010]">
                ML STUDIO
              </span>
              <span className="font-mono text-[10px] font-bold uppercase px-1.5 py-0.5 bg-[#FF5500] text-white border border-[#101010] shadow-[1px_1px_0px_#101010]">
                RUST 2.0
              </span>
            </div>
          </div>
        </div>

        {/* 2-Part Boxy Switcher */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => onTabChange('train')}
            className={`flex items-center gap-2 px-4 py-2 font-mono text-xs font-bold transition-all border-2 border-[#101010] ${
              activeTab === 'train'
                ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                : 'bg-white text-[#101010] hover:bg-[#FAF5ED] shadow-[2px_2px_0px_#101010]'
            }`}
          >
            <Play className={`w-3.5 h-3.5 ${activeTab === 'train' ? 'fill-white' : 'fill-none'}`} />
            <span>01 // TRAIN</span>
          </button>

          <button
            type="button"
            onClick={() => onTabChange('evaluate')}
            className={`flex items-center gap-2 px-4 py-2 font-mono text-xs font-bold transition-all border-2 border-[#101010] ${
              activeTab === 'evaluate'
                ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                : 'bg-white text-[#101010] hover:bg-[#FAF5ED] shadow-[2px_2px_0px_#101010]'
            }`}
          >
            <Search className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>02 // EVALUATE</span>
          </button>
        </div>

        {/* Engine Status */}
        <div className="flex items-center">
          <div className="flex items-center gap-2 px-2.5 py-1 bg-white border-2 border-[#101010] shadow-[2px_2px_0px_#101010] font-mono text-[11px] font-bold">
            <span
              className={`w-2 h-2 border border-[#101010] ${
                backendStatus.online ? 'bg-[#10B981] animate-pulse' : 'bg-[#EF4444]'
              }`}
            />
            <span className="text-[#101010]">
              {backendStatus.checking
                ? 'CHECKING...'
                : backendStatus.online
                ? 'CARGO :8080'
                : 'OFFLINE'}
            </span>
          </div>
        </div>

      </div>
    </header>
  );
};
