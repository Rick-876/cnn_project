import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import './App.css';

// ─── Constants ───────────────────────────────────────────────────────────────
const API_URL = 'http://localhost:8000/predict';

const SAMPLE_QUESTIONS = [
  {
    id: 1,
    title: 'Neural Networks – Backpropagation',
    text: 'Explain the concept of backpropagation in neural networks.',
    maxMarks: 2,
    hint: 'Consider gradient descent, chain rule, and error propagation.',
  },
  {
    id: 2,
    title: 'DHCP Protocol',
    text: 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    maxMarks: 2,
    hint: 'Think about IP address assignment and network configuration.',
  },
  {
    id: 3,
    title: 'Surface Tension',
    text:
      'Look at the 2 pictures of drops on a flat, waterproof surface. The drop on the left is soapy water and the drop on the right is plain water. What causes the plain water to look like it does?',
    maxMarks: 1,
    hint: 'Consider the molecular forces at play on the surface.',
  },
  {
    id: 4,
    title: 'Photosynthesis Process',
    text: 'Describe the process of photosynthesis and explain why it is essential for life on Earth.',
    maxMarks: 2,
    hint: 'Include light reactions, Calvin cycle, and energy production.',
  },
  {
    id: 5,
    title: 'TCP/IP Model Layers',
    text: 'What are the four layers of the TCP/IP model and what is the function of each?',
    maxMarks: 2,
    hint: 'Think about application, transport, internet, and link layers.',
  },
  {
    id: 6,
    title: 'DNA Replication',
    text: 'Explain how DNA replication ensures accurate copying of genetic information.',
    maxMarks: 2,
    hint: 'Consider semi-conservative replication, DNA polymerase, and proofreading.',
  },
  {
    id: 7,
    title: 'Machine Learning Overfitting',
    text: 'What is overfitting in machine learning and how can it be prevented?',
    maxMarks: 2,
    hint: 'Discuss regularization, cross-validation, and model complexity.',
  },
  {
    id: 8,
    title: 'HTTP Protocol',
    text: 'Explain the HTTP protocol and describe the difference between HTTP and HTTPS.',
    maxMarks: 2,
    hint: 'Consider request-response model and security encryption.',
  },
  {
    id: 9,
    title: 'Mitochondrial Function',
    text: 'Describe the role of mitochondria in cellular respiration and energy production.',
    maxMarks: 2,
    hint: 'Include ATP synthesis, electron transport chain, and oxidative phosphorylation.',
  },
  {
    id: 10,
    title: 'Git Version Control',
    text: 'Explain the purpose of version control systems and why they are important in software development.',
    maxMarks: 2,
    hint: 'Consider collaboration, history tracking, and branching.',
  },
  {
    id: 11,
    title: 'Enzyme Catalysis',
    text: 'How do enzymes work as biological catalysts and what role does the active site play?',
    maxMarks: 2,
    hint: 'Think about substrate binding, activation energy, and enzyme specificity.',
  },
  {
    id: 12,
    title: 'Cloud Computing Models',
    text: 'Compare and contrast SaaS, PaaS, and IaaS cloud computing models.',
    maxMarks: 2,
    hint: 'Consider responsibility for infrastructure, platforms, and applications.',
  },
  {
    id: 13,
    title: 'Osmosis Process',
    text: 'What is osmosis and how does the concentration gradient affect water movement across a semipermeable membrane?',
    maxMarks: 2,
    hint: 'Consider solute concentration and water potential.',
  },
  {
    id: 14,
    title: 'Object-Oriented Programming',
    text: 'Explain the concepts of encapsulation, inheritance, and polymorphism in OOP.',
    maxMarks: 3,
    hint: 'Provide examples of how each principle improves code design.',
  },
  {
    id: 15,
    title: 'Photosynthetic Pigments',
    text: 'What are the different types of photosynthetic pigments and their roles in light absorption?',
    maxMarks: 2,
    hint: 'Consider chlorophyll, carotenoids, xanthophyll, and light wavelengths.',
  },
  {
    id: 16,
    title: 'Database Normalization',
    text: 'Explain database normalization and describe the first three normal forms.',
    maxMarks: 3,
    hint: 'Consider redundancy elimination and functional dependencies.',
  },
  {
    id: 17,
    title: 'Protein Synthesis',
    text: 'Describe the steps involved in protein synthesis, including transcription and translation.',
    maxMarks: 3,
    hint: 'Include the role of mRNA, tRNA, ribosomes, and codons.',
  },
  {
    id: 18,
    title: 'API Design Principles',
    text: 'What are the key principles of good REST API design and why is consistency important?',
    maxMarks: 2,
    hint: 'Think about resource naming, HTTP methods, and status codes.',
  },
  {
    id: 19,
    title: 'Cellular Respiration',
    text: 'Compare aerobic and anaerobic respiration in terms of efficiency and products.',
    maxMarks: 2,
    hint: 'Include glycolysis, Krebs cycle, and fermentation.',
  },
  {
    id: 20,
    title: 'Cybersecurity Threats',
    text: 'Explain three common cybersecurity threats and describe methods to mitigate each.',
    maxMarks: 3,
    hint: 'Consider phishing, malware, and social engineering attacks.',
  },
];

const SCORE_DISTRIBUTION_DATA = [
  { label: '0.0', count: 312, color: '#ef4444' },
  { label: '0.33', count: 487, color: '#f97316' },
  { label: '0.5', count: 634, color: '#eab308' },
  { label: '0.67', count: 821, color: '#22c55e' },
  { label: '1.0', count: 1043, color: '#3b82f6' },
];

const MODEL_METRICS = [
  { label: 'Test MSE', value: '0.1403', icon: '📉' },
  { label: 'QWK Score', value: '0.2607', icon: '📊' },
  { label: 'Accuracy', value: '49.6%', icon: '🎯' },
  { label: 'F1 (weighted)', value: '0.493', icon: '⚖️' },
  { label: 'Test Samples', value: '3,038', icon: '🗂️' },
  { label: 'Total Dataset', value: '15,190', icon: '📚' },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function Header() {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-badge">AI · Deep Learning · NLP</div>
        <h1 className="header-title">
          Automatic Short Answer Grading System
        </h1>
        <p className="header-subtitle">
          Convolutional Neural Network (CNN) · ASAG2024 Dataset
        </p>
      </div>
    </header>
  );
}

function QuestionSelector({ selected, onSelect }) {
  return (
    <div className="card question-selector">
      <h2 className="card-title">
        <span className="card-icon">📋</span> Select a Question
      </h2>
      <div className="question-tabs">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q.id}
            className={`question-tab ${selected.id === q.id ? 'active' : ''}`}
            onClick={() => onSelect(q)}
          >
            Q{q.id} · {q.title}
          </button>
        ))}
      </div>
    </div>
  );
}

function QuestionCard({ question }) {
  return (
    <div className="card question-card">
      <div className="question-header">
        <div>
          <span className="question-category">Question</span>
          <h2 className="question-title">{question.title}</h2>
        </div>
        <div className="marks-badge">
          <span className="marks-number">{question.maxMarks}</span>
          <span className="marks-label">mark{question.maxMarks !== 1 ? 's' : ''}</span>
        </div>
      </div>
      <p className="question-text">"{question.text}"</p>
      <div className="question-hint">
        <span className="hint-icon">💡</span>
        <span>{question.hint}</span>
      </div>
    </div>
  );
}

function AnswerInput({ question, onResult, onError }) {
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const MAX_CHARS = 1000;

  // Reset answer when question changes
  useEffect(() => {
    setAnswer('');
  }, [question.id]);

  const handleSubmit = async () => {
    if (!answer.trim()) return;
    setLoading(true);
    onError(null);
    onResult(null);
    try {
      const { data } = await axios.post(
        API_URL,
        { question: question.text, answer },
        { timeout: 10000 },
      );
      onResult(data);
    } catch (err) {
      if (err.code === 'ECONNABORTED' || !err.response) {
        onError(
          'Could not reach the grading server. Make sure the backend is running at ' +
          API_URL,
        );
      } else {
        onError(
          `Server error ${err.response.status}: ${err.response.data?.detail ?? 'Unknown error'}`,
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const remaining = MAX_CHARS - answer.length;
  const pct = Math.min((answer.length / MAX_CHARS) * 100, 100);
  const charColor = remaining < 50 ? '#ef4444' : remaining < 150 ? '#f97316' : '#64748b';

  return (
    <div className="card answer-card">
      <h2 className="card-title">
        <span className="card-icon">✍️</span> Student Answer
      </h2>

      <div className="textarea-wrapper">
        <textarea
          className="answer-textarea"
          placeholder="Type your answer here…"
          value={answer}
          onChange={(e) => setAnswer(e.target.value.slice(0, MAX_CHARS))}
          rows={6}
          disabled={loading}
        />
        <div className="char-counter">
          <div className="char-bar-track">
            <div
              className="char-bar-fill"
              style={{ width: `${pct}%`, backgroundColor: charColor }}
            />
          </div>
          <span style={{ color: charColor }}>
            {answer.length}/{MAX_CHARS} characters
          </span>
        </div>
      </div>

      <button
        className={`submit-btn ${loading ? 'loading' : ''}`}
        onClick={handleSubmit}
        disabled={loading || !answer.trim()}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Grading answer…
          </>
        ) : (
          <>
            <span className="btn-icon">🎓</span>
            Submit for Grading
          </>
        )}
      </button>
    </div>
  );
}

function ResultsCard({ result, maxMarks }) {
  if (!result) return null;

  const { score, confidence, feedback, reference_answer } = result;
  const scorePct = Math.min((score / maxMarks) * 100, 100);

  let scoreClass = 'score-low';
  let scoreLabel = 'Needs Improvement';
  if (scorePct >= 80) { scoreClass = 'score-high'; scoreLabel = 'Excellent'; }
  else if (scorePct >= 50) { scoreClass = 'score-mid'; scoreLabel = 'Satisfactory'; }

  return (
    <div className={`card results-card ${scoreClass}`}>
      <h2 className="card-title">
        <span className="card-icon">📈</span> Grading Result
      </h2>

      <div className="results-grid">
        {/* Score */}
        <div className="result-block score-block">
          <span className="result-label">Predicted Score</span>
          <div className="score-display">
            <span className="score-number">{score}</span>
            <span className="score-denom">/ {maxMarks}</span>
          </div>
          <span className={`score-badge ${scoreClass}`}>{scoreLabel}</span>
        </div>

        {/* Confidence */}
        <div className="result-block confidence-block">
          <span className="result-label">Model Confidence</span>
          <div className="confidence-circle">
            <svg viewBox="0 0 80 80" className="conf-svg">
              <circle cx="40" cy="40" r="34" className="conf-track" />
              <circle
                cx="40" cy="40" r="34"
                className="conf-fill"
                strokeDasharray={`${2 * Math.PI * 34}`}
                strokeDashoffset={`${2 * Math.PI * 34 * (1 - confidence)}`}
              />
            </svg>
            <span className="conf-pct">{Math.round(confidence * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Score progress bar */}
      <div className="score-bar-section">
        <div className="score-bar-track">
          <div
            className={`score-bar-fill ${scoreClass}`}
            style={{ width: `${scorePct}%` }}
          />
        </div>
        <div className="score-bar-labels">
          <span>0</span>
          <span>{maxMarks / 2}</span>
          <span>{maxMarks}</span>
        </div>
      </div>

      {/* Feedback */}
      <div className="feedback-box">
        <span className="feedback-icon">🤖</span>
        <div>
          <p className="feedback-heading">AI Feedback</p>
          <p className="feedback-text">{feedback}</p>
        </div>
      </div>

      {/* Reference Answer */}
      {reference_answer && (
        <div className="reference-box">
          <span className="reference-icon">📚</span>
          <div>
            <p className="reference-heading">Reference Answer</p>
            <p className="reference-text">{reference_answer}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="error-banner">
      <span className="error-icon">⚠️</span>
      <div>
        <p className="error-heading">Grading Unavailable</p>
        <p className="error-text">{message}</p>
      </div>
    </div>
  );
}

function ScoreDistributionChart() {
  return (
    <div className="card insight-card">
      <h2 className="card-title">
        <span className="card-icon">📊</span> Score Distribution (Test Set)
      </h2>
      <p className="insight-sub">
        Distribution of normalized grades across 3,038 test samples
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={SCORE_DISTRIBUTION_DATA} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 12, fill: '#64748b' }} label={{ value: 'Normalized Grade', position: 'insideBottom', offset: -2, fontSize: 12, fill: '#94a3b8' }} />
          <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
            formatter={(v) => [v, 'Answers']}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {SCORE_DISTRIBUTION_DATA.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ModelInsights() {
  return (
    <div className="card insight-card">
      <h2 className="card-title">
        <span className="card-icon">🔬</span> Model Evaluation Insights
      </h2>
      <p className="insight-sub">
        TextCNN trained on ASAG2024 · 80/20 random split · <code>random_state=42</code>
      </p>
      <div className="metrics-grid">
        {MODEL_METRICS.map((m) => (
          <div key={m.label} className="metric-tile">
            <span className="metric-icon">{m.icon}</span>
            <span className="metric-value">{m.value}</span>
            <span className="metric-label">{m.label}</span>
          </div>
        ))}
      </div>
      <div className="architecture-box">
        <p className="arch-heading">Model Architecture</p>
        <div className="arch-pills">
          {['Embedding (300d)', 'Conv2D ×4 (filters: 2,3,4,5)', '128 Filters', 'MaxPool1D', 'Dropout 0.5', 'FC → Score'].map((p) => (
            <span key={p} className="arch-pill">{p}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="app-footer">
      <p>
        Automatic Short Answer Grading System · TextCNN · ASAG2024 Dataset ·
        Built with React &amp; PyTorch
      </p>
    </footer>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedQuestion, setSelectedQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Clear result/error when question changes
  const handleQuestionSelect = (q) => {
    setSelectedQuestion(q);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        {/* Left column */}
        <div className="column-left">
          <QuestionSelector selected={selectedQuestion} onSelect={handleQuestionSelect} />
          <QuestionCard question={selectedQuestion} />
          <AnswerInput
            question={selectedQuestion}
            onResult={setResult}
            onError={setError}
          />
          <ErrorBanner message={error} />
          <ResultsCard result={result} maxMarks={selectedQuestion.maxMarks} />
        </div>

        {/* Right column */}
        <div className="column-right">
          <ScoreDistributionChart />
          <ModelInsights />
        </div>
      </main>

      <Footer />
    </div>
  );
}
