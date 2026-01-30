"use client";

import { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Bar, Line, Scatter } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface VisualizationData {
  status: string;
  task_type: string;
  project_name: string;
  predictions: {
    actual: number[];
    predicted: number[];
    probabilities?: number[][];
    residuals?: number[];
  };
  feature_importance?: {
    features: string[];
    importance: number[];
  };
  model_comparison: {
    models: string[];
    scores: {
      [key: string]: number[];
    };
  };
}

interface ModelVisualizationsProps {
  projectName: string;
  taskType: string;
}

export default function ModelVisualizations({ projectName, taskType }: ModelVisualizationsProps) {
  const [vizData, setVizData] = useState<VisualizationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  console.log('ModelVisualizations rendered with:', { projectName, taskType });

  useEffect(() => {
    const fetchVisualizationData = async () => {
      try {
        setLoading(true);
        console.log('Fetching visualizations for project:', projectName);
        const response = await fetch(`/api/automl/visualizations/${projectName}`);
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.status === 'success') {
          setVizData(data);
        } else {
          setError(data.error || 'Failed to load visualization data');
        }
      } catch (err) {
        console.error('Visualization fetch error:', err);
        setError('Failed to communicate with backend');
      } finally {
        setLoading(false);
      }
    };

    if (projectName) {
      fetchVisualizationData();
    }
  }, [projectName]);

  if (loading) {
    return (
      <div style={{ padding: 20, textAlign: 'center' }}>
        <p>Loading visualizations...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 20, textAlign: 'center', color: '#ef4444' }}>
        <p>❌ {error}</p>
      </div>
    );
  }

  if (!vizData) return null;

  // Model Comparison Chart
  const modelComparisonData = {
    labels: vizData.model_comparison.models,
    datasets: Object.keys(vizData.model_comparison.scores).map((metric, idx) => ({
      label: metric.toUpperCase(),
      data: vizData.model_comparison.scores[metric],
      backgroundColor: [
        'rgba(99, 102, 241, 0.7)',
        'rgba(16, 185, 129, 0.7)',
        'rgba(245, 158, 11, 0.7)',
        'rgba(239, 68, 68, 0.7)',
      ][idx % 4],
      borderColor: [
        'rgb(99, 102, 241)',
        'rgb(16, 185, 129)',
        'rgb(245, 158, 11)',
        'rgb(239, 68, 68)',
      ][idx % 4],
      borderWidth: 2,
    })),
  };

  const modelComparisonOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
        font: { size: 16, weight: 'bold' },
      },
    },
    scales: {
      y: {
        beginAtZero: taskType === 'classification',
      },
    },
  };

  // Actual vs Predicted Scatter Plot
  const scatterData = {
    datasets: [
      {
        label: 'Predictions',
        data: vizData.predictions.actual.map((actual, idx) => ({
          x: actual,
          y: vizData.predictions.predicted[idx],
        })),
        backgroundColor: 'rgba(99, 102, 241, 0.6)',
        borderColor: 'rgb(99, 102, 241)',
        pointRadius: 4,
      },
      ...(taskType === 'regression' ? [{
        label: 'Perfect Prediction',
        data: [
          { x: Math.min(...vizData.predictions.actual), y: Math.min(...vizData.predictions.actual) },
          { x: Math.max(...vizData.predictions.actual), y: Math.max(...vizData.predictions.actual) },
        ],
        borderColor: 'rgba(239, 68, 68, 0.8)',
        borderWidth: 2,
        pointRadius: 0,
        type: 'line' as const,
        borderDash: [5, 5],
      }] : []),
    ],
  };

  const scatterOptions: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Actual vs Predicted Values',
        font: { size: 16, weight: 'bold' },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Actual Values',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Predicted Values',
        },
      },
    },
  };

  // Feature Importance Chart
  const featureImportanceData = vizData.feature_importance ? {
    labels: vizData.feature_importance.features,
    datasets: [
      {
        label: 'Importance',
        data: vizData.feature_importance.importance,
        backgroundColor: 'rgba(16, 185, 129, 0.7)',
        borderColor: 'rgb(16, 185, 129)',
        borderWidth: 2,
      },
    ],
  } : null;

  const featureImportanceOptions: ChartOptions<'bar'> = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Top Feature Importance',
        font: { size: 16, weight: 'bold' },
      },
    },
  };

  // Residuals Distribution (for regression)
  const residualsData = vizData.predictions.residuals ? {
    labels: vizData.predictions.residuals.map((_, idx) => idx.toString()),
    datasets: [
      {
        label: 'Residuals',
        data: vizData.predictions.residuals,
        backgroundColor: 'rgba(245, 158, 11, 0.6)',
        borderColor: 'rgb(245, 158, 11)',
        borderWidth: 1,
      },
    ],
  } : null;

  const residualsOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Residuals Distribution',
        font: { size: 16, weight: 'bold' },
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Residual (Actual - Predicted)',
        },
      },
      x: {
        display: false,
      },
    },
  };

  return (
    <div style={{ marginTop: 24 }}>
      <div
        style={{
          padding: 20,
          borderRadius: 14,
          border: '1px solid var(--border)',
          background: 'rgba(15, 23, 42, 0.03)',
        }}
      >
        <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 20 }}>
          📊 Model Visualizations
        </h2>

        {/* Model Comparison */}
        <div style={{ marginBottom: 32, height: 400 }}>
          <Bar data={modelComparisonData} options={modelComparisonOptions} />
        </div>

        {/* Actual vs Predicted */}
        <div style={{ marginBottom: 32, height: 400 }}>
          <Scatter data={scatterData} options={scatterOptions} />
        </div>

        {/* Feature Importance */}
        {featureImportanceData && (
          <div style={{ marginBottom: 32, height: 400 }}>
            <Bar data={featureImportanceData} options={featureImportanceOptions} />
          </div>
        )}

        {/* Residuals (Regression only) */}
        {residualsData && taskType === 'regression' && (
          <div style={{ marginBottom: 32, height: 400 }}>
            <Bar data={residualsData} options={residualsOptions} />
          </div>
        )}

        {/* Error Statistics */}
        {taskType === 'regression' && vizData.predictions.residuals && (
          <div style={{ marginTop: 24, padding: 16, borderRadius: 10, background: 'rgba(99, 102, 241, 0.1)' }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Error Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
              <div>
                <p style={{ fontSize: 11, color: 'var(--muted)' }}>Mean Error</p>
                <p style={{ fontSize: 16, fontWeight: 700 }}>
                  {(vizData.predictions.residuals.reduce((a, b) => a + b, 0) / vizData.predictions.residuals.length).toFixed(4)}
                </p>
              </div>
              <div>
                <p style={{ fontSize: 11, color: 'var(--muted)' }}>Std Deviation</p>
                <p style={{ fontSize: 16, fontWeight: 700 }}>
                  {Math.sqrt(
                    vizData.predictions.residuals.reduce((a, b) => a + b * b, 0) / vizData.predictions.residuals.length
                  ).toFixed(4)}
                </p>
              </div>
              <div>
                <p style={{ fontSize: 11, color: 'var(--muted)' }}>Max Error</p>
                <p style={{ fontSize: 16, fontWeight: 700 }}>
                  {Math.max(...vizData.predictions.residuals.map(Math.abs)).toFixed(4)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
