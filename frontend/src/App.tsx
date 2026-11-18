import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { TrainModelFlow } from './components/TrainModelFlow';
import { EvaluateModelFlow } from './components/EvaluateModelFlow';
import { Cpu, Terminal, Shield } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'train' | 'evaluate'>('train');

  return (
    <div className="pixel-grid-bg min-h-screen flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Flow Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
        {activeTab === 'train' ? <TrainModelFlow /> : <EvaluateModelFlow />}
      </main>

      {/* Minimal Boxy Footer */}
      <footer className="border-t-2 border-[#101010] bg-[#FAF5ED] py-4 text-xs font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-[#101010]">
          <div className="flex items-center gap-3">
            <span className="font-bold bg-[#101010] text-white px-2 py-0.5 border border-[#101010]">
              CORE
            </span>
            <span className="font-bold">RUST ML ENGINE // CARGO AXUM (:8080)</span>
          </div>

          <div className="flex items-center gap-2 text-[11px] font-semibold text-[#555]">
            <span className="px-1.5 py-0.5 bg-white border border-[#101010]">.PT</span>
            <span className="px-1.5 py-0.5 bg-white border border-[#101010]">.PKT</span>
            <span className="px-1.5 py-0.5 bg-white border border-[#101010]">.JSON</span>
            <span className="px-1.5 py-0.5 bg-white border border-[#101010]">.BIN</span>
            <span className="ml-2 font-bold text-[#101010]">// 100% LOCAL COMPUTE</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
