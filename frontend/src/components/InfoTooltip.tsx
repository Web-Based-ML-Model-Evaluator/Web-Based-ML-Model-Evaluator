import React, { useState, useRef, useEffect } from 'react';

export interface InfoTooltipProps {
  title?: string;
  content: string | React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  style?: React.CSSProperties;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  title,
  content,
  position = 'top',
  style,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside or pressing Escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsOpen(!isOpen);
  };

  // Popup positioning styles
  const getPopoverStyle = (): React.CSSProperties => {
    const base: React.CSSProperties = {
      position: 'absolute',
      zIndex: 1050,
      width: '260px',
      backgroundColor: '#2D241E',
      color: '#FBF8F4',
      borderRadius: '8px',
      padding: '10px 14px',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.22), 0 2px 6px rgba(0, 0, 0, 0.12)',
      fontSize: '12px',
      lineHeight: '1.45',
      textAlign: 'left',
      pointerEvents: 'auto',
      border: '1px solid rgba(255, 255, 255, 0.1)',
    };

    switch (position) {
      case 'bottom':
        return {
          ...base,
          top: 'calc(100% + 8px)',
          left: '50%',
          transform: 'translateX(-50%)',
        };
      case 'left':
        return {
          ...base,
          right: 'calc(100% + 8px)',
          top: '50%',
          transform: 'translateY(-50%)',
        };
      case 'right':
        return {
          ...base,
          left: 'calc(100% + 8px)',
          top: '50%',
          transform: 'translateY(-50%)',
        };
      case 'top':
      default:
        return {
          ...base,
          bottom: 'calc(100% + 8px)',
          left: '50%',
          transform: 'translateX(-50%)',
        };
    }
  };

  return (
    <div
      ref={containerRef}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        position: 'relative',
        marginLeft: '6px',
        verticalAlign: 'middle',
        ...style,
      }}
    >
      <button
        type="button"
        onClick={toggle}
        onMouseEnter={() => setIsOpen(true)}
        aria-label="More information"
        title="Click for explanation"
        style={{
          width: '16px',
          height: '16px',
          borderRadius: '50%',
          border: isOpen ? '1px solid #D45A1E' : '1px solid #C4B5A5',
          backgroundColor: isOpen ? '#FAF0E8' : 'transparent',
          color: isOpen ? '#D45A1E' : '#8A7D71',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '10px',
          fontWeight: 700,
          fontFamily: 'serif',
          fontStyle: 'italic',
          cursor: 'pointer',
          padding: 0,
          lineHeight: 1,
          transition: 'all 0.15s ease-in-out',
          outline: 'none',
        }}
      >
        i
      </button>

      {isOpen && (
        <div style={getPopoverStyle()} onClick={(e) => e.stopPropagation()}>
          {title && (
            <div
              style={{
                fontWeight: 700,
                fontSize: '12px',
                color: '#F7A072',
                marginBottom: '4px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span>ⓘ</span>
              <span>{title}</span>
            </div>
          )}
          <div style={{ color: '#E8E1D9', fontSize: '11.5px' }}>{content}</div>
        </div>
      )}
    </div>
  );
};
