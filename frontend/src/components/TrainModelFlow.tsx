import React, { useState } from 'react';
import {
  Upload,
  Play,
  Check,
  AlertCircle,
  Download,
  RotateCcw,
  Sparkles,
  Sliders,
  Layers,
  ChevronRight,
  Database,
  FileCode,
} from 'lucide-react';
import {
  BackendApi,
  DatasetAnalysis,
  DatasetQuestionsResponse,
  AlgorithmDefinition,
  TrainResponse,
} from '../services/backendApi';
import { createSampleFile } from '../services/sampleData';

export const TrainModelFlow: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [trainingLoading, setTrainingLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [questions, setQuestions] = useState<DatasetQuestionsResponse | null>(null);

  const [taskType, setTaskType] = useState<'classification' | 'regression' | 'clustering'>('classification');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('random_forest_classifier');
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({
    n_trees: 10,
    max_depth: 5,
    k: 3,
    epochs: 50,
    learning_rate: 0.01,
    alpha: 1.0,
    eps: 0.5,
    min_samples: 3,
    num_clusters: 3,
  });
  const [testSplitRatio, setTestSplitRatio] = useState<number>(0.2);
  const [stratifiedSplit, setStratifiedSplit] = useState<boolean>(true);

  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setErrorMessage(null);
    setTrainResult(null);
    try {
      const dataAnalysis = await BackendApi.uploadDataset(file);
      setAnalysis(dataAnalysis);

      const q = await BackendApi.getDatasetQuestions(dataAnalysis.dataset_id);
      setQuestions(q);

      const recTask = (q.recommended_task as 'classification' | 'regression' | 'clustering') || 'classification';
      setTaskType(recTask);

      if (recTask === 'classification' && q.classification_targets.length > 0) {
        setTargetColumn(q.classification_targets[0].name);
        setSelectedAlgorithm('random_forest_classifier');
      } else if (recTask === 'regression' && q.regression_targets.length > 0) {
        setTargetColumn(q.regression_targets[0].name);
        setSelectedAlgorithm('linear_regression');
      } else if (recTask === 'clustering') {
        setTargetColumn('');
        setSelectedAlgorithm('kmeans');
      }

      const features = q.candidate_features
        .map((c) => c.name)
        .filter((name) => name !== q.classification_targets[0]?.name && name !== q.regression_targets[0]?.name);
      setFeatureColumns(features);

      setCurrentStep(2);
    } catch (err: any) {
      setErrorMessage(err.message || 'Dataset upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleTaskTypeChange = (newType: 'classification' | 'regression' | 'clustering') => {
    setTaskType(newType);
    if (!questions) return;

    if (newType === 'classification') {
      const target = questions.classification_targets[0]?.name || '';
      setTargetColumn(target);
      setSelectedAlgorithm('random_forest_classifier');
      setFeatureColumns(
        questions.candidate_features.map((c) => c.name).filter((name) => name !== target)
      );
    } else if (newType === 'regression') {
      const target = questions.regression_targets[0]?.name || '';
      setTargetColumn(target);
      setSelectedAlgorithm('linear_regression');
      setFeatureColumns(
        questions.candidate_features.map((c) => c.name).filter((name) => name !== target)
      );
    } else {
      setTargetColumn('');
      setSelectedAlgorithm('kmeans');
      setFeatureColumns(questions.candidate_features.map((c) => c.name));
    }
  };

  const currentAlgorithms: AlgorithmDefinition[] =
    questions?.algorithms.filter((a) => a.task_type === taskType) || [];

  const currentAlgoDef = currentAlgorithms.find((a) => a.id === selectedAlgorithm) || currentAlgorithms[0];

  const handleTrain = async () => {
    if (!analysis) return;
    setTrainingLoading(true);
    setErrorMessage(null);
    try {
      const res = await BackendApi.trainModel({
        dataset_id: analysis.dataset_id,
        task_type: taskType,
        algorithm: selectedAlgorithm,
        target_column: taskType === 'clustering' ? null : targetColumn,
        feature_columns: featureColumns,
        test_split_ratio: testSplitRatio,
        stratified_split: stratifiedSplit,
        hyperparameters,
      });
      setTrainResult(res);
      setCurrentStep(3);
    } catch (err: any) {
      setErrorMessage(err.message || 'Model training failed.');
    } finally {
      setTrainingLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Boxy Step Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-3 bg-white border-2 border-[#101010] shadow-[3px_3px_0px_#101010]">
        <div className="flex items-center gap-2 font-mono text-xs font-bold">
          <button
            onClick={() => setCurrentStep(1)}
            className={`px-3 py-1.5 border border-[#101010] transition-all ${
              currentStep === 1
                ? 'bg-[#FF5500] text-white shadow-[2px_2px_0px_#101010]'
                : analysis
                ? 'bg-[#FAF5ED] text-[#101010] hover:bg-white'
                : 'bg-gray-100 text-gray-400'
            }`}
          >
            [01] INGEST
          </button>

          <span>//</span>

          <button
            disabled={!analysis}
            onClick={() => analysis && setCurrentStep(2)}
            className={`px-3 py-1.5 border border-[#101010] transition-all ${
              currentStep === 2
                ? 'bg-[#FF5500] text-white shadow-[2px_2px_0px_#101010]'
                : currentStep > 2
                ? 'bg-[#FAF5ED] text-[#101010] hover:bg-white'
                : 'bg-gray-100 text-gray-400 opacity-60'
            }`}
          >
            [02] CONFIGURE
          </button>

          <span>//</span>

          <button
            disabled={!trainResult}
            onClick={() => trainResult && setCurrentStep(3)}
            className={`px-3 py-1.5 border border-[#101010] transition-all ${
              currentStep === 3
                ? 'bg-[#FF5500] text-white shadow-[2px_2px_0px_#101010]'
                : trainResult
                ? 'bg-[#FAF5ED] text-[#101010] hover:bg-white'
                : 'bg-gray-100 text-gray-400 opacity-60'
            }`}
          >
            [03] EVAL & EXPORT
          </button>
        </div>

        {analysis && (
          <div className="font-mono text-xs font-bold flex items-center gap-2 bg-[#FAF5ED] px-2.5 py-1 border border-[#101010]">
            <Database className="w-3.5 h-3.5 text-[#FF5500]" />
            <span>{analysis.filename}</span>
            <span className="text-gray-500">[{analysis.total_rows}R x {analysis.total_columns}C]</span>
          </div>
        )}
      </div>

      {/* Error Callout */}
      {errorMessage && (
        <div className="p-3 bg-red-100 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] font-mono text-xs font-bold text-red-900 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
          <span>ERROR // {errorMessage}</span>
        </div>
      )}

      {/* STEP 1: INGEST */}
      {currentStep === 1 && (
        <div className="space-y-6">
          {/* Main Dropzone */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-10 text-center relative group">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="w-16 h-16 mx-auto mb-4 bg-[#FF5500] text-white border-2 border-[#101010] shadow-[3px_3px_0px_#101010] flex items-center justify-center group-hover:-translate-y-1 transition-transform">
              <Upload className="w-8 h-8" />
            </div>
            <h3 className="font-mono text-lg font-black text-[#101010] uppercase tracking-wide mb-1">
              DROP CSV DATASET HERE
            </h3>
            <p className="font-mono text-xs text-gray-500 mb-6">
              [ TABULAR DATA // AUTO SCHEMA DETECTION // ZERO LEAKAGE ]
            </p>
            <div className="inline-block px-4 py-2 bg-[#101010] text-white font-mono text-xs font-bold border-2 border-[#101010] shadow-[2px_2px_0px_#FF5500]">
              SELECT LOCAL FILE
            </div>
          </div>

          {/* Quick Presets */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3 flex items-center gap-2">
              <span>// 1-CLICK BENCHMARK DATASETS</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                type="button"
                onClick={() => handleFileUpload(createSampleFile('iris'))}
                className="text-left p-3.5 bg-[#FAF5ED] hover:bg-[#FFF2EB] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] hover:-translate-x-0.5 hover:-translate-y-0.5 transition-all group"
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono font-black text-sm text-[#101010] group-hover:text-[#FF5500]">
                    IRIS.CSV
                  </span>
                  <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 bg-[#101010] text-white">
                    CLASSIFY
                  </span>
                </div>
                <p className="font-mono text-[11px] text-gray-600">
                  30 rows · 4 features · 3 species classes
                </p>
              </button>

              <button
                type="button"
                onClick={() => handleFileUpload(createSampleFile('housing'))}
                className="text-left p-3.5 bg-[#FAF5ED] hover:bg-[#FFF2EB] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] hover:-translate-x-0.5 hover:-translate-y-0.5 transition-all group"
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono font-black text-sm text-[#101010] group-hover:text-[#FF5500]">
                    HOUSING.CSV
                  </span>
                  <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 bg-[#101010] text-white">
                    REGRESS
                  </span>
                </div>
                <p className="font-mono text-[11px] text-gray-600">
                  25 rows · continuous target: price
                </p>
              </button>

              <button
                type="button"
                onClick={() => handleFileUpload(createSampleFile('customer'))}
                className="text-left p-3.5 bg-[#FAF5ED] hover:bg-[#FFF2EB] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] hover:-translate-x-0.5 hover:-translate-y-0.5 transition-all group"
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono font-black text-sm text-[#101010] group-hover:text-[#FF5500]">
                    CUSTOMER.CSV
                  </span>
                  <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 bg-[#101010] text-white">
                    CLUSTER
                  </span>
                </div>
                <p className="font-mono text-[11px] text-gray-600">
                  25 rows · income/spending centroids
                </p>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* STEP 2: CONFIGURE */}
      {currentStep === 2 && analysis && questions && (
        <div className="space-y-6">
          {/* Section 1: Task Type */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
              [01] SELECT TASK PARADIGM
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                { id: 'classification', title: 'CLASSIFICATION', desc: 'Predict discrete category labels' },
                { id: 'regression', title: 'REGRESSION', desc: 'Predict continuous numerical scalars' },
                { id: 'clustering', title: 'CLUSTERING', desc: 'Unsupervised centroid partitioning' },
              ].map((t) => {
                const isSelected = taskType === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => handleTaskTypeChange(t.id as any)}
                    className={`text-left p-4 border-2 border-[#101010] transition-all ${
                      isSelected
                        ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                        : 'bg-[#FAF5ED] text-[#101010] hover:bg-white shadow-[2px_2px_0px_#101010]'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-1 font-mono font-black text-sm">
                      <span>{t.title}</span>
                      {isSelected && <Check className="w-4 h-4 stroke-[3]" />}
                    </div>
                    <p className={`font-mono text-[11px] ${isSelected ? 'text-white/90' : 'text-gray-600'}`}>
                      {t.desc}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Target & Features */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
              [02] TARGET & INPUT FEATURES
            </div>

            {taskType !== 'clustering' ? (
              <div className="mb-5">
                <label className="block font-mono text-xs font-bold uppercase text-[#101010] mb-2">
                  TARGET COLUMN // ({taskType.toUpperCase()})
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => {
                    const col = e.target.value;
                    setTargetColumn(col);
                    setFeatureColumns((prev) => prev.filter((f) => f !== col));
                  }}
                  className="w-full max-w-md p-2.5 font-mono text-xs font-bold bg-[#FAF5ED] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] focus:outline-none focus:border-[#FF5500]"
                >
                  {taskType === 'classification' &&
                    questions.classification_targets.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({c.unique_count} classes)
                      </option>
                    ))}
                  {taskType === 'regression' &&
                    questions.regression_targets.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} (range: {c.min_val}..{c.max_val})
                      </option>
                    ))}
                </select>
              </div>
            ) : (
              <div className="mb-5 p-3 bg-[#FAF5ED] border-2 border-[#101010] font-mono text-xs font-bold text-[#101010]">
                CLUSTERING // Unsupervised clustering operates on all selected features without labels.
              </div>
            )}

            <div>
              <div className="flex justify-between items-center mb-2 font-mono text-xs font-bold">
                <span className="uppercase text-[#101010]">
                  INPUT FEATURES ({featureColumns.length} ACTIVE)
                </span>
                <button
                  type="button"
                  onClick={() => {
                    const all = questions.candidate_features
                      .filter((c) => c.name !== targetColumn)
                      .map((c) => c.name);
                    setFeatureColumns(featureColumns.length === all.length ? [] : all);
                  }}
                  className="text-[10px] underline uppercase"
                >
                  TOGGLE ALL
                </button>
              </div>

              <div className="flex flex-wrap gap-2">
                {questions.candidate_features
                  .filter((col) => col.name !== targetColumn)
                  .map((col) => {
                    const active = featureColumns.includes(col.name);
                    return (
                      <button
                        key={col.name}
                        type="button"
                        onClick={() => {
                          if (active) {
                            setFeatureColumns(featureColumns.filter((f) => f !== col.name));
                          } else {
                            setFeatureColumns([...featureColumns, col.name]);
                          }
                        }}
                        className={`px-3 py-1.5 font-mono text-xs font-bold border-2 border-[#101010] transition-all ${
                          active
                            ? 'bg-[#101010] text-white shadow-[2px_2px_0px_#FF5500]'
                            : 'bg-white text-gray-500 hover:text-[#101010]'
                        }`}
                      >
                        {col.name}
                      </button>
                    );
                  })}
              </div>
            </div>
          </div>

          {/* Section 3: Algorithm Architecture */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
              [03] CHOOSE ALGORITHM ARCHITECTURE
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              {currentAlgorithms.map((algo) => {
                const isSelected = selectedAlgorithm === algo.id;
                return (
                  <button
                    key={algo.id}
                    type="button"
                    onClick={() => setSelectedAlgorithm(algo.id)}
                    className={`text-left p-3 border-2 border-[#101010] transition-all flex flex-col justify-between ${
                      isSelected
                        ? 'bg-[#FF5500] text-white shadow-[3px_3px_0px_#101010] -translate-x-0.5 -translate-y-0.5'
                        : 'bg-[#FAF5ED] text-[#101010] hover:bg-white shadow-[2px_2px_0px_#101010]'
                    }`}
                  >
                    <div>
                      <div className="font-mono font-black text-xs uppercase mb-1">
                        {algo.name}
                      </div>
                      <p className={`font-mono text-[10px] line-clamp-2 ${isSelected ? 'text-white/90' : 'text-gray-600'}`}>
                        {algo.description}
                      </p>
                    </div>
                    <div className="mt-2 pt-1.5 border-t border-[#101010]/30 font-mono text-[9px] uppercase tracking-wider">
                      {algo.complexity}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 4: Hyperparameters */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="flex justify-between items-center mb-3">
              <div className="font-mono text-xs font-bold uppercase text-gray-500">
                [04] HYPERPARAMETERS // {currentAlgoDef.name}
              </div>
              <span className="font-mono text-[10px] px-2 py-0.5 bg-[#FAF5ED] border border-[#101010] font-bold">
                {currentAlgoDef.hyperparameters.length} PARAMETERS
              </span>
            </div>

            {/* Requirement Guide */}
            <div className="p-3 bg-[#FAF5ED] border-2 border-[#101010] mb-4">
              <div className="font-mono text-[11px] font-bold uppercase text-[#FF5500] mb-1">
                // SPECIFICATION REQUIREMENTS
              </div>
              {currentAlgoDef.hyperparameters.length === 0 ? (
                <p className="font-mono text-xs text-gray-600">
                  Closed-form / Analytic optimization. No hyperparameter tuning required.
                </p>
              ) : (
                <div className="space-y-1">
                  {currentAlgoDef.hyperparameters.map((spec) => (
                    <div key={spec.name} className="font-mono text-[11px] text-[#101010]">
                      <span className="font-bold uppercase text-[#FF5500]">{spec.label}:</span>{' '}
                      <span>{spec.requirement_text}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Range Controls */}
            {currentAlgoDef.hyperparameters.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {currentAlgoDef.hyperparameters.map((spec) => {
                  const val = hyperparameters[spec.name] ?? spec.default_value;
                  return (
                    <div key={spec.name} className="p-3 bg-[#FAF5ED] border-2 border-[#101010]">
                      <div className="flex justify-between items-center font-mono text-xs font-bold mb-2">
                        <span className="uppercase">{spec.label}</span>
                        <span className="px-2 py-0.5 bg-[#101010] text-white">{val}</span>
                      </div>
                      <input
                        type="range"
                        min={spec.min ?? 1}
                        max={spec.max ?? 100}
                        step={spec.step ?? 1}
                        value={val}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          setHyperparameters({ ...hyperparameters, [spec.name]: v });
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Section 5: Split Configuration */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
              [05] TRAIN / TEST SPLIT
            </div>

            <div className="max-w-md space-y-3">
              <div className="flex justify-between items-center font-mono text-xs font-bold">
                <span>RATIO</span>
                <span className="px-2 py-0.5 bg-[#FF5500] text-white">
                  {((1 - testSplitRatio) * 100).toFixed(0)}% TRAIN // {(testSplitRatio * 100).toFixed(0)}% TEST
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.50"
                step="0.05"
                value={testSplitRatio}
                onChange={(e) => setTestSplitRatio(parseFloat(e.target.value))}
              />

              {taskType === 'classification' && (
                <label className="flex items-center gap-2 pt-2 cursor-pointer font-mono text-xs font-bold text-[#101010]">
                  <input
                    type="checkbox"
                    checked={stratifiedSplit}
                    onChange={(e) => setStratifiedSplit(e.target.checked)}
                  />
                  <span>STRATIFIED SAMPLING (MAINTAINS EXACT CLASS BALANCE)</span>
                </label>
              )}
            </div>
          </div>

          {/* Action Button */}
          <div className="flex justify-end">
            <button
              type="button"
              disabled={trainingLoading || featureColumns.length === 0}
              onClick={handleTrain}
              className="boxy-btn px-8 py-3.5 font-mono text-sm uppercase flex items-center gap-2"
            >
              {trainingLoading ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>TRAINING ON RUST ENGINE...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>EXECUTE TRAINING</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* STEP 3: SCORES & DOWNLOAD */}
      {currentStep === 3 && trainResult && (
        <div className="space-y-6">
          {/* Header Banner */}
          <div className="bg-[#101010] text-white border-2 border-[#101010] shadow-[4px_4px_0px_#FF5500] p-5 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="font-mono text-[10px] text-[#FF5500] font-black uppercase mb-1">
                // RUST EXECUTION COMPLETE
              </div>
              <h3 className="font-mono text-lg font-black tracking-wide">
                MODEL: {trainResult.model_id}
              </h3>
              <div className="font-mono text-xs text-gray-400 mt-1 flex flex-wrap gap-3">
                <span>ALGO: {trainResult.algorithm}</span>
                <span>•</span>
                <span>TRAIN TIME: {trainResult.train_time_ms}ms</span>
                <span>•</span>
                <span>SAMPLES: {trainResult.train_samples} TR / {trainResult.test_samples} TE</span>
              </div>
            </div>

            <button
              onClick={() => setCurrentStep(2)}
              className="boxy-btn-white px-4 py-2 font-mono text-xs flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>RETRAIN</span>
            </button>
          </div>

          {/* Model Artifact Downloads */}
          <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-5">
            <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
              [ ARTIFACT EXPORTS // LOCAL DOWNLOAD ]
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { format: 'pt', name: '.PT', desc: 'PyTorch Container' },
                { format: 'pkt', name: '.PKT', desc: 'SmartCore Binary' },
                { format: 'json', name: '.JSON', desc: 'Weights Schema' },
                { format: 'bin', name: '.BIN', desc: 'Raw Executable' },
              ].map((f) => (
                <a
                  key={f.format}
                  href={BackendApi.getDownloadUrl(trainResult.model_id, f.format)}
                  download
                  className="p-3 bg-[#FAF5ED] hover:bg-[#FFF2EB] border-2 border-[#101010] shadow-[2px_2px_0px_#101010] text-center flex flex-col items-center gap-1 hover:-translate-x-0.5 hover:-translate-y-0.5 transition-all group"
                >
                  <Download className="w-4 h-4 text-[#FF5500]" />
                  <span className="font-mono font-black text-sm text-[#101010] group-hover:text-[#FF5500]">
                    {f.name}
                  </span>
                  <span className="font-mono text-[10px] text-gray-500">{f.desc}</span>
                </a>
              ))}
            </div>
          </div>

          {/* Metric Dashboard: Classification */}
          {trainResult.task_type === 'classification' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#FF5500] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">ACCURACY</div>
                  <div className="font-mono text-3xl font-black text-[#FF5500] my-1">
                    {((trainResult.metrics.accuracy || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">CORRECT SAMPLES</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">F1 SCORE</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(trainResult.metrics.f1_score || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">HARMONIC MEAN</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">PRECISION</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(trainResult.metrics.precision || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">PPV RATIO</div>
                </div>

                <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                  <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">RECALL</div>
                  <div className="font-mono text-3xl font-black text-[#101010] my-1">
                    {(trainResult.metrics.recall || 0).toFixed(3)}
                  </div>
                  <div className="font-mono text-[9px] text-gray-400">TRUE POSITIVE</div>
                </div>
              </div>

              {/* Per Class Table & Confusion Matrix */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-4">
                  <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
                    // CLASS BREAKDOWN
                  </div>
                  <table className="w-full font-mono text-xs border-collapse">
                    <thead>
                      <tr className="border-b-2 border-[#101010] text-gray-500">
                        <th className="py-1 text-left">LABEL</th>
                        <th className="py-1 text-right">PREC</th>
                        <th className="py-1 text-right">REC</th>
                        <th className="py-1 text-right">F1</th>
                        <th className="py-1 text-right">N</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y border-b border-[#101010]">
                      {(trainResult.metrics.per_class || []).map((row: any) => (
                        <tr key={row.class}>
                          <td className="py-1.5 font-bold text-[#101010]">{row.class}</td>
                          <td className="py-1.5 text-right font-bold text-[#FF5500]">{row.precision}</td>
                          <td className="py-1.5 text-right font-bold">{row.recall}</td>
                          <td className="py-1.5 text-right font-bold">{row.f1_score}</td>
                          <td className="py-1.5 text-right text-gray-500">{row.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {trainResult.confusion_matrix && (
                  <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-4">
                    <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
                      // CONFUSION MATRIX
                    </div>
                    <table className="w-full font-mono text-xs text-center border-collapse">
                      <thead>
                        <tr>
                          <th className="p-1"></th>
                          {trainResult.confusion_matrix.labels.map((lbl) => (
                            <th key={lbl} className="p-1 font-bold text-gray-500 truncate max-w-[60px]">
                              P:{lbl}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {trainResult.confusion_matrix.matrix.map((row, i) => (
                          <tr key={i}>
                            <td className="p-1 font-bold text-gray-500 text-left truncate max-w-[60px]">
                              A:{trainResult.confusion_matrix!.labels[i]}
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
            </div>
          )}

          {/* Metric Dashboard: Regression */}
          {trainResult.task_type === 'regression' && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#FF5500] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">R² SCORE</div>
                <div className="font-mono text-3xl font-black text-[#FF5500] my-1">
                  {(trainResult.metrics.r2_score || 0).toFixed(4)}
                </div>
                <div className="font-mono text-[9px] text-gray-400">EXPLAINED VARIANCE</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">MAE</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(trainResult.metrics.mean_absolute_error || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">MEAN ABSOLUTE</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">RMSE</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(trainResult.metrics.root_mean_squared_error || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">ROOT MEAN SQUARED</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">MAX RESIDUAL</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(trainResult.metrics.max_residual || 0).toLocaleString()}
                </div>
                <div className="font-mono text-[9px] text-gray-400">LARGEST DEVIATION</div>
              </div>
            </div>
          )}

          {/* Metric Dashboard: Clustering */}
          {trainResult.task_type === 'clustering' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#FF5500] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">SILHOUETTE</div>
                <div className="font-mono text-3xl font-black text-[#FF5500] my-1">
                  {(trainResult.metrics.silhouette_score || 0).toFixed(3)}
                </div>
                <div className="font-mono text-[9px] text-gray-400">SEPARATION QUALITY</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">DAVIES-BOULDIN</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {(trainResult.metrics.davies_bouldin_index || 0).toFixed(3)}
                </div>
                <div className="font-mono text-[9px] text-gray-400">SIMILARITY INDEX</div>
              </div>

              <div className="bg-white p-4 border-2 border-[#101010] shadow-[3px_3px_0px_#101010] text-center">
                <div className="font-mono text-[10px] font-bold text-gray-500 uppercase">CLUSTERS</div>
                <div className="font-mono text-3xl font-black text-[#101010] my-1">
                  {trainResult.metrics.num_clusters}
                </div>
                <div className="font-mono text-[9px] text-gray-400">CENTROIDS DETECTED</div>
              </div>
            </div>
          )}

          {/* Sample Predictions Table */}
          {trainResult.sample_predictions && trainResult.sample_predictions.length > 0 && (
            <div className="bg-white border-2 border-[#101010] shadow-[4px_4px_0px_#101010] p-4">
              <div className="font-mono text-xs font-bold uppercase text-gray-500 mb-3">
                // SAMPLE PREDICTIONS PREVIEW
              </div>
              <div className="overflow-x-auto">
                <table className="w-full font-mono text-xs">
                  <thead>
                    <tr className="border-b-2 border-[#101010] text-gray-500 text-left">
                      <th className="py-1">#</th>
                      <th className="py-1">GROUND TRUTH</th>
                      <th className="py-1">PREDICTION</th>
                      <th className="py-1 text-right">STATUS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {trainResult.sample_predictions.map((p, idx) => (
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
