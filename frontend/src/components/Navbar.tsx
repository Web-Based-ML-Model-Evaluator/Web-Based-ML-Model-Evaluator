import React from 'react';

interface NavbarProps {
  currentView: 'home' | 'train' | 'evaluate';
  onNavigate: (view: 'home' | 'train' | 'evaluate') => void;
  gpuDeviceName?: string;
  isGpuActive?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentView,
  onNavigate,
  gpuDeviceName = 'Hardware WebGPU Device',
  isGpuActive = true,
}) => {
  return (
    <header
      style={{
        width: '100%',
        padding: '24px 48px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxSizing: 'border-box',
      }}
    >
      {/* Brand Logo & Name (Matching Screenshot) */}
      <div
        onClick={() => onNavigate('home')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        {/* Isometric 3-Stacked Orange Diamonds Logo */}
        <svg
          width="32"
          height="32"
          viewBox="0 0 32 32"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Top Layer */}
          <path
            d="M16 3L28 9L16 15L4 9L16 3Z"
            fill="#E2632B"
          />
          {/* Middle Layer */}
          <path
            d="M4 14.5L16 20.5L28 14.5L28 16.5L16 22.5L4 16.5V14.5Z"
            fill="#E2632B"
            opacity="0.9"
          />
          {/* Bottom Layer */}
          <path
            d="M4 20L16 26L28 20V22L16 28L4 22V20Z"
            fill="#C94F1A"
          />
        </svg>

        <span
          style={{
            fontSize: '22px',
            fontWeight: 800,
            color: '#1C1917',
            letterSpacing: '-0.3px',
            fontFamily: "var(--font-sans, 'Inter', sans-serif)",
          }}
        >
          ML Lab
        </span>
      </div>

      {/* Center Nav Pills (Smooth navigation between Home, Train, Evaluate) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          backgroundColor: 'rgba(255, 255, 255, 0.5)',
          padding: '4px',
          borderRadius: '24px',
          border: '1px solid #EBE1D5',
        }}
      >
        <button
          type="button"
          onClick={() => onNavigate('home')}
          style={{
            padding: '6px 16px',
            borderRadius: '20px',
            border: 'none',
            backgroundColor: currentView === 'home' ? '#FFFFFF' : 'transparent',
            color: currentView === 'home' ? '#D45A1E' : '#72665C',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            boxShadow: currentView === 'home' ? '0 2px 6px rgba(0,0,0,0.05)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          Home
        </button>
        <button
          type="button"
          onClick={() => onNavigate('train')}
          style={{
            padding: '6px 16px',
            borderRadius: '20px',
            border: 'none',
            backgroundColor: currentView === 'train' ? '#FFFFFF' : 'transparent',
            color: currentView === 'train' ? '#D45A1E' : '#72665C',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            boxShadow: currentView === 'train' ? '0 2px 6px rgba(0,0,0,0.05)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          Train a Model
        </button>
        <button
          type="button"
          onClick={() => onNavigate('evaluate')}
          style={{
            padding: '6px 16px',
            borderRadius: '20px',
            border: 'none',
            backgroundColor: currentView === 'evaluate' ? '#FFFFFF' : 'transparent',
            color: currentView === 'evaluate' ? '#D45A1E' : '#72665C',
            fontWeight: 600,
            fontSize: '13px',
            cursor: 'pointer',
            boxShadow: currentView === 'evaluate' ? '0 2px 6px rgba(0,0,0,0.05)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          Evaluate a Model
        </button>
      </div>

      {/* Right Controls: GPU Badge & Profile Avatar (Matching Screenshot) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* GPU Status Pill */}
        <div
          title={`Compute Hardware: ${gpuDeviceName}`}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            borderRadius: '16px',
            backgroundColor: isGpuActive ? '#F0FDF4' : '#F8FAFC',
            border: `1px solid ${isGpuActive ? '#BBF7D0' : '#E2E8F0'}`,
            fontSize: '11px',
            fontWeight: 600,
            color: isGpuActive ? '#166534' : '#64748B',
          }}
        >
          <span
            style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              backgroundColor: isGpuActive ? '#22C55E' : '#94A3B8',
              boxShadow: isGpuActive ? '0 0 6px #22C55E' : 'none',
            }}
          />
          <span>GPU/VRAM Accelerated</span>
        </div>

        {/* Circular User Avatar Button (From Screenshot) */}
        <button
          type="button"
          className="user-avatar-btn"
          title="User Account"
          aria-label="User Account"
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            border: '1.5px solid #E6DDD1',
            backgroundColor: '#FFFFFF',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            padding: 0,
            boxShadow: '0 2px 5px rgba(0,0,0,0.04)',
          }}
        >
          {/* Simple minimalist line-art user silhouette */}
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#38302A"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </button>
      </div>
    </header>
  );
};
