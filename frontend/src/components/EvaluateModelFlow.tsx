import React, { useState } from 'react';
import {
  FileCode,
  FileSpreadsheet,
  AlertCircle,
  Play,
} from 'lucide-react';
import { BackendApi, EvaluateResponse } from '../services/backendApi';

const SUPPORTED_EXTS = ['.PT', '.PTH', '.PT2', '.PTE', '.PTL', '.PKT', '.PKL', '.BIN', '.JSON', '.ONNX'];

export const EvaluateModelFlow: React.FC = () => {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);

  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [algorithm, setAlgorithm] = useState<string>('random_forest_classifier');
  const [targetColumn, setTargetColumn] = useState<string>('');

  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [evalResult, setEvalResult] = useState<EvaluateResponse | null>(null);

  const handleRunEvaluation = async () => {
    if (!modelFile || !datasetFile) {
      setErrorMessage('PROVIDE BOTH MODEL FILE AND TEST CSV DATASET');
      return;
    }

    setEvaluating(true);
    setErrorMessage(null);
    setEvalResult(null);

    try {
      const res = await BackendApi.evaluateModelFile(
        modelFile,
        datasetFile,
        taskType,
        algorithm,
        targetColumn || undefined
      );
      setEvalResult(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Evaluation failed.');
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Boxy Header */}
      <div className="p-4 bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-mono text-xs font-bold uppercase text-gray-500">
            // UNIVERSAL MODEL EVALUATOR
          </div>
          <h2 className="font-mono text-xl font-black text-[#101010] tracking-tight">
            EVALUATE PRE-TRAINED ARTIFACTS
          </h2>
        </div>

        <div className="flex flex-wrap gap-1 font-mono text-[10px] font-bold">
          {SUPPORTED_EXTS.map((ext) => (
            <span key={ext} className="px-1.5 py-0.5 bg-[#FAF5ED] border border-[#101010]">
              {ext}
            </span>
          ))}
        </div>
      </div>

      {/* Error Callout */}
      {errorMessage && (
        <div className="p-3 bg-red-100 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] font-mono text-xs font-bold text-red-900 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
          <span>ERROR // {errorMessage}</span>
        </div>
      )}

      {/* Dual Upload Columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Model File Upload */}
        <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5 space-y-3">
          <div className="flex justify-between items-center font-mono text-xs font-bold">
            <span className="text-gray-500">[01] MODEL WEIGHTS FILE</span>
            {modelFile && (
              <span className="px-1.5 py-0.5 bg-[#101010] text-[#10B981]">LOADED</span>
            )}
          </div>

          <div className="border-2 border-dashed border-[#101010] p-6 text-center bg-[#FAF5ED] relative group hover:bg-[#FFF2EB] transition-colors">
            <input
              type="file"
              onChange={(e) => e.target.files?.[0] && setModelFile(e.target.files[0])}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            {modelFile ? (
              <div className="space-y-1 font-mono">
                <div className="font-bold text-sm text-[#101010]">{modelFile.name}</div>
                <div className="text-[11px] text-gray-500">{(modelFile.size / 1024).toFixed(1)} KB</div>
              </div>
            ) : (
              <div className="space-y-1 font-mono">
                <FileCode className="w-8 h-8 text-[#FF5500] mx-auto mb-2" />
                <div className="font-bold text-xs text-[#101010] uppercase">
                  DROP MODEL FILE (.PT, .PKT, .JSON, .BIN...)
                </div>
                <div className="text-[10px] text-gray-500">CLICK TO BROWSE LOCAL STORAGE</div>
              </div>
            )}
          </div>
        </div>

        {/* Test Dataset Upload */}
        <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5 space-y-3">
          <div className="flex justify-between items-center font-mono text-xs font-bold">
            <span className="text-gray-500">[02] TEST DATASET CSV</span>
            {datasetFile && (
              <span className="px-1.5 py-0.5 bg-[#101010] text-[#10B981]">LOADED</span>
            )}
          </div>

          <div className="border-2 border-dashed border-[#101010] p-6 text-center bg-[#FAF5ED] relative group hover:bg-[#FFF2EB] transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files?.[0] && setDatasetFile(e.target.files[0])}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            {datasetFile ? (
              <div className="space-y-1 font-mono">
                <div className="font-bold text-sm text-[#101010]">{datasetFile.name}</div>
                <div className="text-[11px] text-gray-500">{(datasetFile.size / 1024).toFixed(1)} KB</div>
              </div>
            ) : (
              <div className="space-y-1 font-mono">
                <FileSpreadsheet className="w-8 h-8 text-[#FF5500] mx-auto mb-2" />
                <div className="font-bold text-xs text-[#101010] uppercase">
                  DROP TEST CSV (GROUND TRUTH LABELS)
                </div>
                <div className="text-[10px] text-gray-500">CLICK TO BROWSE LOCAL CSV</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Architecture Combination */}
      <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5 space-y-4">
        <div className="font-mono text-xs font-bold uppercase text-gray-500">
          [03] SPECIFY MODEL ARCHITECTURE COMBINATION
        </div>

        {/* Task Switcher */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => {
              setTaskType('classification');
              setAlgorithm('random_forest_classifier');
            }}
            className={`p-3 font-mono text-xs font-black uppercase border-2 border-[#101010] transition-all ${
              taskType === 'classification'
                ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                : 'bg-[#FAF5ED] text-[#101010] hover:bg-white shadow-[2px_2px_0px_#101010]'
            }`}
          >
            CLASSIFICATION MODEL
          </button>

          <button
            type="button"
            onClick={() => {
              setTaskType('regression');
              setAlgorithm('linear_regression');
            }}
            className={`p-3 font-mono text-xs font-black uppercase border-2 border-[#101010] transition-all ${
              taskType === 'regression'
                ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                : 'bg-[#FAF5ED] text-[#101010] hover:bg-white shadow-[2px_2px_0px_#101010]'
            }`}
          >
            REGRESSION MODEL
          </button>
        </div>

        {/* Algorithm Grid */}
        <div>
          <div className="font-mono text-[10px] text-gray-500 font-bold mb-2 uppercase">
            // SELECT EXACT MODEL ARCHITECTURE:
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(taskType === 'classification'
              ? [
                  { id: 'logistic_regression', name: 'LOGISTIC REGRESSION' },
                  { id: 'decision_tree_classifier', name: 'DECISION TREE' },
                  { id: 'random_forest_classifier', name: 'RANDOM FOREST' },
                  { id: 'knn_classifier', name: 'K-NEAREST NEIGHBORS' },
                  { id: 'svm_classifier', name: 'SVM (SVC)' },
                  { id: 'naive_bayes', name: 'GAUSSIAN NAIVE BAYES' },
                  { id: 'neural_network', name: 'NEURAL NETWORK (MLP)' },
                  { id: 'cnn_classifier', name: '1D CONVOLUTIONAL (CNN)' },
                ]
              : [
                  { id: 'linear_regression', name: 'LINEAR REGRESSION (OLS)' },
                  { id: 'ridge_regression', name: 'RIDGE REGRESSION (L2)' },
                  { id: 'lasso_regression', name: 'LASSO REGRESSION (L1)' },
                  { id: 'decision_tree_regressor', name: 'DECISION TREE REGRESSOR' },
                  { id: 'random_forest_regressor', name: 'RANDOM FOREST REGRESSOR' },
                  { id: 'knn_regressor', name: 'KNN REGRESSOR' },
                  { id: 'svr', name: 'SUPPORT VECTOR REGRESSOR' },
                  { id: 'neural_network_regressor', name: 'NEURAL NETWORK (MLP)' },
                ]
            ).map((item) => {
              const isSelected = algorithm === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setAlgorithm(item.id)}
                  className={`p-2.5 font-mono text-xs font-bold text-left border-2 border-[#101010] transition-all ${
                    isSelected
                      ? 'bg-[#101010] text-[#FF5500] shadow-[2px_2px_0px_#FF5500]'
                      : 'bg-[#FAF5ED] text-[#101010] hover:bg-white'
                  }`}
                >
                  {item.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Target Column */}
        <div className="pt-2">
          <label className="block font-mono text-xs font-bold uppercase text-[#101010] mb-1.5">
            GROUND TRUTH TARGET COLUMN
          </label>
          <input
            type="text"
            placeholder="e.g. species, price (or leave empty to auto-detect last column)"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="w-full max-w-md p-2.5 font-mono text-xs font-bold bg-[#FAF5ED] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] focus:outline-none focus:border-[#FF5500]"
          />
        </div>

        {/* Action Button */}
        <div className="flex justify-end pt-3">
          <button
            type="button"
            disabled={evaluating || !modelFile || !datasetFile}
            onClick={handleRunEvaluation}
            className="boxy-btn px-8 py-3.5 font-mono text-sm uppercase flex items-center gap-2"
          >
            {evaluating ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>EVALUATING ON RUST ENGINE...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>EVALUATE ARTIFACT</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results Dashboard */}
      {evalResult && (
        <div className="space-y-6">
          {/* Header Banner */}
          <div className="bg-[#101010] text-white border-2 border-[#101010] shadow-[4px_4px_0px_#FF5500] p-5">
            <div className="font-mono text-[10px] text-[#FF5500] font-black uppercase mb-1">
              // EVALUATION REPORT
            </div>
            <h3 className="font-mono text-lg font-black tracking-wide">
              {evalResult.model_filename} [{evalResult.model_format.toUpperCase()}]
            </h3>
            <div className="font-mono text-xs text-gray-400 mt-1 flex flex-wrap gap-3">
              <span>TEST SET: {evalResult.dataset_filename}</span>
              <span>•</span>
              <span>SAMPLES EVALUATED: {evalResult.total_evaluated_samples}</span>
              <span>•</span>
              <span>TIME: {evalResult.evaluation_time_ms}ms</span>
            </div>
          </div>

          {/* Classification Metrics */}
          {evalResult.task_type === 'classification' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#FF5500] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">ACCURACY</div>
                  <div className="font-mono text-3xl font-black text-[#FF5500] my-1">
                    {((evalResult.metrics.accuracy || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">CORRECT MATCHES</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">F1 SCORE</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(evalResult.metrics.f1_score || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">HARMONIC MEAN</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">PRECISION</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(evalResult.metrics.precision || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">PPV RATIO</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">RECALL</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(evalResult.metrics.recall || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">TRUE POSITIVE</div>
                </div>
              </div>

              {/* Confusion Matrix */}
              {evalResult.confusion_matrix && (
                <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-4">
                  <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
                    // CONFUSION MATRIX
                  </div>
                  <table className="w-full font-mono text-xs text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-1"></th>
                        {evalResult.confusion_matrix.labels.map((lbl) => (
                          <th key={lbl} className="p-1 font-bold text-gray-500">
                            PRED {lbl}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {evalResult.confusion_matrix.matrix.map((row, i) => (
                        <tr key={i}>
                          <td className="p-1 font-bold text-gray-500 text-left">
                            ACT {evalResult.confusion_matrix!.labels[i]}
                          </td>
                          {row.map((val, j) => (
                            <td
                              key={j}
                              className={`p-2 font-bold border border-[#101010] ${
                                i === j
                                  ? val > 0
                                    ? 'bg-[#101010] text-[#FF5500]'
                                    : 'bg-[#FAF5ED]'
                                  : val > 0
                                  ? 'bg-red-500 text-white'
                                  : 'bg-white text-gray-300'
                              }`}
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* Regression Metrics */}
          {evalResult.task_type === 'regression' && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#FF5500] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">R² SCORE</div>
                <div className="font-mono text-3xl font-black text-[#FF5500] my-1">
                  {(evalResult.metrics.r2_score || 0).toFixed(4)}
                </div>
                <div className="font-mono text-[9px] text-gray-400">EXPLAINED VARIANCE</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">MAE</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(evalResult.metrics.mean_absolute_error || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">MEAN ABSOLUTE</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">RMSE</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(evalResult.metrics.root_mean_squared_error || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">ROOT MEAN SQUARED</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">MAX RESIDUAL</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(evalResult.metrics.max_residual || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">LARGEST DEVIATION</div>
              </div>
            </div>
          )}

          {/* Sample Predictions Table */}
          {evalResult.sample_predictions && evalResult.sample_predictions.length > 0 && (
            <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-4">
              <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
                // EVALUATION PREDICTIONS
              </div>
              <div className="overflow-x-auto">
                <table className="w-full font-mono text-xs">
                  <thead>
                    <tr className="border-b-2 border-[#101010] text-gray-500 text-left">
                      <th className="py-1">#</th>
                      <th className="py-1">GROUND TRUTH</th>
                      <th className="py-1">EVAL PREDICTION</th>
                      <th className="py-1 text-right">STATUS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {evalResult.sample_predictions.map((p, idx) => (
                      <tr key={idx}>
                        <td className="py-1.5 text-gray-400">#{idx + 1}</td>
                        <td className="py-1.5 font-bold text-[#101010]">{String(p.actual)}</td>
                        <td className="py-1.5 font-bold text-[#101010]">{String(p.predicted)}</td>
                        <td className="py-1.5 text-right">
                          {p.correct !== undefined ? (
                            p.correct ? (
                              <span className="px-1.5 py-0.5 bg-[#101010] text-[#10B981] font-bold">
                                MATCH
                              </span>
                            ) : (
                              <span className="px-1.5 py-0.5 bg-red-600 text-white font-bold">
                                ERROR
                              </span>
                            )
                          ) : (
                            <span className="text-gray-500">{p.error}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
