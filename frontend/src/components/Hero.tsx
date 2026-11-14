import React from 'react';

interface HeroProps {
  onSelectAction: (action: 'train' | 'evaluate') => void;
}

export const Hero: React.FC<HeroProps> = ({ onSelectAction }) => {
  return (
    <section
      style={{
        maxWidth: '1000px',
        margin: '40px auto 60px',
        padding: '0 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
      }}
    >
      {/* Decorative Framed Area with Corner Brackets (Exact Screenshot Match) */}
      <div
        style={{
          position: 'relative',
          padding: '16px 36px',
          display: 'inline-block',
          marginBottom: '16px',
        }}
      >
        {/* Top-Left Corner Bracket ┌ */}
        <div
          style={{
            position: 'absolute',
            top: '0',
            left: '0',
            width: '28px',
            height: '28px',
            borderTop: '3px solid #D4A579',
            borderLeft: '3px solid #D4A579',
          }}
        />

        {/* Bottom-Right Corner Bracket ┘ */}
        <div
          style={{
            position: 'absolute',
            bottom: '0',
            right: '0',
            width: '28px',
            height: '28px',
            borderBottom: '3px solid #D4A579',
            borderRight: '3px solid #D4A579',
          }}
        />

        {/* Main Serif Headline */}
        <h1
          style={{
            margin: 0,
            fontFamily: "var(--font-serif, 'Playfair Display', Georgia, serif)",
            fontSize: '52px',
            lineHeight: 1.15,
            fontWeight: 700,
            color: '#221C18',
            letterSpacing: '-0.5px',
          }}
        >
          Train and Evaluate
          <br />
          <span>Your </span>
          <span
            style={{
              position: 'relative',
              display: 'inline-block',
              color: '#D45A1E',
            }}
          >
            Models
            {/* Hand-Drawn Brush-Stroke Underline (Screenshot Match) */}
            <svg
              className="brush-underline"
              viewBox="0 0 170 18"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              style={{
                position: 'absolute',
                left: '-2px',
                bottom: '-10px',
                width: '104%',
                height: '14px',
                overflow: 'visible',
              }}
            >
              <path
                d="M3 10C35 15.5 105 16 166 7.5C125 12 55 13 12 8.5"
                stroke="#D45A1E"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </h1>
      </div>

      {/* Subtitle */}
      <p
        style={{
          maxWidth: '580px',
          margin: '20px auto 48px',
          fontSize: '17px',
          lineHeight: 1.6,
          color: '#73675E',
          fontWeight: 400,
        }}
      >
        A simple platform to train custom models on your dataset and evaluate existing models with the same data.
      </p>

      {/* Action Cards Container (2 Side-by-Side Cards Matching Screenshot) */}
      <div
        style={{
          width: '100%',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '28px',
          maxWidth: '860px',
        }}
      >
        {/* Card 1: Train a Model (Active Border from Screenshot) */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectAction('train')}
          onKeyDown={(e) => e.key === 'Enter' && onSelectAction('train')}
          className="action-card active-border"
          style={{
            backgroundColor: 'rgba(255, 254, 251, 0.75)',
            border: '1.5px solid #E59A68',
            borderRadius: '24px',
            padding: '28px',
            textAlign: 'left',
            boxShadow: '0 4px 20px rgba(212, 90, 30, 0.08)',
            cursor: 'pointer',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            minHeight: '190px',
          }}
        >
          <div>
            {/* Peach Badge with Orange Stacked Layers Icon */}
            <div
              style={{
                width: '46px',
                height: '46px',
                borderRadius: '12px',
                backgroundColor: '#FDECE0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M12 2L21 6.5L12 11L3 6.5L12 2Z"
                  stroke="#D45A1E"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M3 12L12 16.5L21 12"
                  stroke="#D45A1E"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M3 17.5L12 22L21 17.5"
                  stroke="#D45A1E"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            {/* Title */}
            <h2
              style={{
                margin: '0 0 8px',
                fontSize: '20px',
                fontWeight: 700,
                color: '#1C1917',
              }}
            >
              Train a Model
            </h2>

            {/* Description */}
            <p
              style={{
                margin: 0,
                fontSize: '14px',
                lineHeight: 1.5,
                color: '#73675E',
                maxWidth: '260px',
              }}
            >
              Upload your dataset, choose a model, and train it.
            </p>
          </div>

          {/* Bottom-Right Arrow Button */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              marginTop: '16px',
            }}
          >
            <div
              className="action-card-btn"
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '10px',
                backgroundColor: '#D45A1E',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#FFFFFF',
              }}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14" />
                <path d="m12 5 7 7-7 7" />
              </svg>
            </div>
          </div>
        </div>

        {/* Card 2: Evaluate a Model (Subtle Border from Screenshot) */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onSelectAction('evaluate')}
          onKeyDown={(e) => e.key === 'Enter' && onSelectAction('evaluate')}
          className="action-card subtle-border"
          style={{
            backgroundColor: 'rgba(255, 254, 251, 0.75)',
            border: '1.5px solid #E8DDCF',
            borderRadius: '24px',
            padding: '28px',
            textAlign: 'left',
            boxShadow: '0 4px 16px rgba(100, 60, 20, 0.03)',
            cursor: 'pointer',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            minHeight: '190px',
          }}
        >
          <div>
            {/* Soft Amber Badge with Bar Chart Icon */}
            <div
              style={{
                width: '46px',
                height: '46px',
                borderRadius: '12px',
                backgroundColor: '#FBF0E2',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#C46E31"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="20" x2="18" y2="10" />
                <line x1="12" y1="20" x2="12" y2="4" />
                <line x1="6" y1="20" x2="6" y2="14" />
              </svg>
            </div>

            {/* Title */}
            <h2
              style={{
                margin: '0 0 8px',
                fontSize: '20px',
                fontWeight: 700,
                color: '#1C1917',
              }}
            >
              Evaluate a Model
            </h2>

            {/* Description */}
            <p
              style={{
                margin: 0,
                fontSize: '14px',
                lineHeight: 1.5,
                color: '#73675E',
                maxWidth: '260px',
              }}
            >
              Upload your model and dataset to evaluate its performance.
            </p>
          </div>

          {/* Bottom-Right Arrow Button */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              marginTop: '16px',
            }}
          >
            <div
              className="action-card-btn"
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '10px',
                backgroundColor: '#D45A1E',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#FFFFFF',
              }}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14" />
                <path d="m12 5 7 7-7 7" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
