import { useState, useEffect } from 'react';
import {
  IonApp,
  IonRouterOutlet,
  IonTabs,
  IonTabBar,
  IonTabButton,
  IonIcon,
  IonLabel,
  IonPage,
  IonContent,
  IonHeader,
  IonToolbar,
  IonTitle,
  IonList,
  IonItem,
  IonToggle,
  IonInput,
  IonButton,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
  IonProgressBar,
  IonBadge,
  IonSpinner,
  setupIonicReact,
  IonRefresher,
  IonRefresherContent,
} from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route, Redirect } from 'react-router-dom';
import { home, settings, playCircle, barChart } from 'ionicons/icons';
import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';
import './App.css';

setupIonicReact();

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:12000';

interface Model { name: string; display_name: string; provider: string; }
interface Dataset { name: string; item_count: number; has_ground_truth: boolean; }
interface EvaluationRun { id: string; name: string; status: string; dataset_name: string; }
interface Summary { models?: Record<string, { successful: number; metrics: Record<string, { mean: number }> }>; }

const api = {
  async health() { return fetch(`${API_BASE}/health`).then(r => r.json()); },
  async getInfo() { return fetch(`${API_BASE}/info`).then(r => r.json()); },
  async getDatasets(): Promise<Dataset[]> { return fetch(`${API_BASE}/datasets`).then(r => r.json()); },
  async getEvaluations(): Promise<{ runs: EvaluationRun[] }> { return fetch(`${API_BASE}/evaluations`).then(r => r.json()); },
  async getStatus(runId: string) { return fetch(`${API_BASE}/evaluations/${runId}/status`).then(r => r.json()); },
  async getSummary(runId: string): Promise<Summary> { return fetch(`${API_BASE}/evaluations/${runId}/summary`).then(r => r.json()); },
  async startEvaluation(datasetName: string, models: string[]) {
    return fetch(`${API_BASE}/evaluations`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_name: datasetName, models }),
    }).then(r => r.json());
  },
};

const HomePage = () => {
  const [info, setInfo] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    api.health().then(() => setConnected(true)).catch(() => setConnected(false));
    api.getInfo().then(setInfo);
  }, []);
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>AI Eval Pipeline</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <div className="page-content">
          <div className={`status-badge ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </div>
          {info && (
            <>
              <h2>🤖 {info.service}</h2>
              <IonCard>
                <IonCardHeader><IonCardTitle>Available Models</IonCardTitle></IonCardHeader>
                <IonCardContent>
                  <div className="model-grid">
                    {info.available_models?.map((m: Model) => (
                      <span key={m.name} className="model-chip">{m.display_name}</span>
                    ))}
                  </div>
                </IonCardContent>
              </IonCard>
              <IonButton expand="block" routerLink="/evaluate" className="action-btn">🚀 Start Evaluation</IonButton>
              <IonButton expand="block" routerLink="/results" className="action-btn secondary">📊 View Results</IonButton>
            </>
          )}
        </div>
      </IonContent>
    </IonPage>
  );
};

const ModelsPage = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  useEffect(() => {
    api.getInfo().then((info) => {
      setModels(info.available_models || []);
      const all = info.available_models?.map((m: Model) => m.name) || [];
      setSelected(new Set(all));
      localStorage.setItem('selected_models', JSON.stringify(all));
    });
  }, []);
  const toggleModel = (name: string) => {
    const ns = new Set(selected);
    ns.has(name) ? ns.delete(name) : ns.add(name);
    setSelected(ns);
    localStorage.setItem('selected_models', JSON.stringify([...ns]));
  };
  const grouped = models.reduce((acc: Record<string, Model[]>, m) => {
    (acc[m.provider] ||= []).push(m); return acc;
  }, {});
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>Select Models</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <div className="page-content">
          {Object.entries(grouped).map(([provider, ms]) => (
            <IonCard key={provider}>
              <IonCardHeader><IonCardTitle>{provider.toUpperCase()}</IonCardTitle></IonCardHeader>
              <IonCardContent>
                {ms.map((m) => (
                  <IonItem key={m.name}>
                    <IonLabel>{m.display_name}</IonLabel>
                    <IonToggle checked={selected.has(m.name)} onIonChange={() => toggleModel(m.name)} slot="end" />
                  </IonItem>
                ))}
              </IonCardContent>
            </IonCard>
          ))}
          <p className="selected-count">{selected.size} models selected</p>
        </div>
      </IonContent>
    </IonPage>
  );
};

const DatasetsPage = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const refresh = async () => { setLoading(true); setDatasets(await api.getDatasets()); setLoading(false); };
  useEffect(() => { refresh(); }, []);
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>Datasets</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <IonRefresher slot="fixed" onIonRefresh={(e: CustomEvent) => refresh().then(() => e.detail.complete())}>
          <IonRefresherContent />
        </IonRefresher>
        <div className="page-content">
          {loading ? <IonSpinner /> : datasets.length === 0 ? (
            <p className="empty-state">No datasets. Upload via web dashboard.</p>
          ) : datasets.map((ds) => (
            <IonCard key={ds.name}>
              <IonCardHeader><IonCardTitle>{ds.name}</IonCardTitle></IonCardHeader>
              <IonCardContent>
                <p>📊 {ds.item_count} items</p>
                <p>Ground Truth: {ds.has_ground_truth ? '✅' : '❌'}</p>
              </IonCardContent>
            </IonCard>
          ))}
        </div>
      </IonContent>
    </IonPage>
  );
};

const EvaluatePage = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDs, setSelectedDs] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [status, setStatus] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  useEffect(() => { api.getDatasets().then(setDatasets); setSelectedModels(JSON.parse(localStorage.getItem('selected_models') || '[]')); }, []);
  useEffect(() => {
    if (!runId) return;
    const poll = async () => {
      const s = await api.getStatus(runId);
      setStatus(s);
      if (s.status === 'completed' || s.status === 'failed') setRunning(false);
    };
    poll();
    const i = setInterval(poll, 3000);
    return () => clearInterval(i);
  }, [runId]);
  const startEval = async () => {
    if (!selectedDs || !selectedModels.length) return;
    setRunning(true);
    const r = await api.startEvaluation(selectedDs, selectedModels);
    setRunId(r.run_id);
  };
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>Run Evaluation</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <div className="page-content">
          <IonCard>
            <IonCardHeader><IonCardTitle>Configuration</IonCardTitle></IonCardHeader>
            <IonCardContent>
              <IonList>
                <IonItem>
                  <IonLabel position="stacked">Dataset</IonLabel>
                  <select value={selectedDs} onChange={(e) => setSelectedDs(e.target.value)} className="native-select">
                    <option value="">Select...</option>
                    {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
                  </select>
                </IonItem>
                <IonItem><IonLabel>Models: {selectedModels.length} selected</IonLabel></IonItem>
              </IonList>
              <IonButton expand="block" onClick={startEval} disabled={!selectedDs || !selectedModels.length || running}>
                {running ? '⏳ Running...' : '🚀 Start Evaluation'}
              </IonButton>
            </IonCardContent>
          </IonCard>
          {status && (
            <IonCard>
              <IonCardHeader><IonCardTitle>Status</IonCardTitle></IonCardHeader>
              <IonCardContent>
                <IonBadge color={status.status === 'completed' ? 'success' : status.status === 'failed' ? 'danger' : 'primary'}>{status.status}</IonBadge>
                <IonProgressBar value={status.progress / 100} />
                <p>Progress: {status.progress?.toFixed(1)}%</p>
                <p>Success: {status.successful} | Failed: {status.failed}</p>
                {status.status === 'completed' && <IonButton routerLink="/results" expand="block">View Results →</IonButton>}
              </IonCardContent>
            </IonCard>
          )}
        </div>
      </IonContent>
    </IonPage>
  );
};

const ResultsPage = () => {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [summary, setSummary] = useState<Summary | null>(null);
  useEffect(() => {
    api.getEvaluations().then((d) => {
      const completed = d.runs?.filter((r) => r.status === 'completed') || [];
      setRuns(completed);
      if (completed.length) setSelectedRun(completed[0].id);
    });
  }, []);
  useEffect(() => { if (selectedRun) api.getSummary(selectedRun).then(setSummary); }, [selectedRun]);
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>Results</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <div className="page-content">
          {!runs.length ? (
            <p className="empty-state">No completed evaluations.</p>
          ) : (
            <>
              <IonCard>
                <IonCardContent>
                  <select value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)} className="native-select">
                    {runs.map((r) => <option key={r.id} value={r.id}>{r.name || r.id.slice(0, 8)}</option>)}
                  </select>
                </IonCardContent>
              </IonCard>
              {summary?.models && (
                <IonCard>
                  <IonCardHeader><IonCardTitle>📊 Comparison</IonCardTitle></IonCardHeader>
                  <IonCardContent>
                    <table className="results-table">
                      <thead><tr><th>Model</th><th>Accuracy</th><th>Latency</th></tr></thead>
                      <tbody>
                        {Object.entries(summary.models).map(([name, data]) => (
                          <tr key={name}>
                            <td>{name}</td>
                            <td>{data.metrics.accuracy?.mean?.toFixed(3) || 'N/A'}</td>
                            <td>{data.metrics.latency_ms?.mean?.toFixed(0) || 'N/A'}ms</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </IonCardContent>
                </IonCard>
              )}
              {summary?.models && (
                <IonCard>
                  <IonCardHeader><IonCardTitle>🏆 Winner</IonCardTitle></IonCardHeader>
                  <IonCardContent>
                    {(() => {
                      const winner = Object.entries(summary.models!).sort((a, b) =>
                        (b[1].metrics.composite_score?.mean || 0) - (a[1].metrics.composite_score?.mean || 0)
                      )[0];
                      return <div><h3>🥇 {winner[0]}</h3><p>Score: {winner[1].metrics.composite_score?.mean?.toFixed(3)}</p></div>;
                    })()}
                  </IonCardContent>
                </IonCard>
              )}
            </>
          )}
        </div>
      </IonContent>
    </IonPage>
  );
};

const SettingsPage = () => {
  const [apiUrl, setApiUrl] = useState(localStorage.getItem('api_url') || API_BASE);
  const [openaiKey, setOpenaiKey] = useState(localStorage.getItem('openai_key') || '');
  const [anthropicKey, setAnthropicKey] = useState(localStorage.getItem('anthropic_key') || '');
  const [groqKey, setGroqKey] = useState(localStorage.getItem('groq_key') || '');
  const save = () => {
    localStorage.setItem('api_url', apiUrl);
    localStorage.setItem('openai_key', openaiKey);
    localStorage.setItem('anthropic_key', anthropicKey);
    localStorage.setItem('groq_key', groqKey);
  };
  return (
    <IonPage>
      <IonHeader><IonToolbar><IonTitle>Settings</IonTitle></IonToolbar></IonHeader>
      <IonContent>
        <div className="page-content">
          <IonCard>
            <IonCardHeader><IonCardTitle>API Connection</IonCardTitle></IonCardHeader>
            <IonCardContent>
              <IonList>
                <IonItem><IonLabel position="stacked">API URL</IonLabel><IonInput value={apiUrl} onIonChange={(e) => setApiUrl(e.detail.value!)} /></IonItem>
              </IonList>
            </IonCardContent>
          </IonCard>
          <IonCard>
            <IonCardHeader><IonCardTitle>API Keys</IonCardTitle></IonCardHeader>
            <IonCardContent>
              <IonList>
                <IonItem><IonLabel position="stacked">OpenAI</IonLabel><IonInput type="password" value={openaiKey} onIonChange={(e) => setOpenaiKey(e.detail.value!)} /></IonItem>
                <IonItem><IonLabel position="stacked">Anthropic</IonLabel><IonInput type="password" value={anthropicKey} onIonChange={(e) => setAnthropicKey(e.detail.value!)} /></IonItem>
                <IonItem><IonLabel position="stacked">Groq</IonLabel><IonInput type="password" value={groqKey} onIonChange={(e) => setGroqKey(e.detail.value!)} /></IonItem>
              </IonList>
              <IonButton expand="block" onClick={save}>Save</IonButton>
            </IonCardContent>
          </IonCard>
        </div>
      </IonContent>
    </IonPage>
  );
};

const App = () => (
  <IonApp>
    <IonReactRouter>
      <IonTabs>
        <IonRouterOutlet>
          <Route exact path="/home" component={HomePage} />
          <Route exact path="/models" component={ModelsPage} />
          <Route exact path="/datasets" component={DatasetsPage} />
          <Route exact path="/evaluate" component={EvaluatePage} />
          <Route exact path="/results" component={ResultsPage} />
          <Route exact path="/settings" component={SettingsPage} />
          <Route exact path="/"><Redirect to="/home" /></Route>
        </IonRouterOutlet>
        <IonTabBar slot="bottom">
          <IonTabButton tab="home" href="/home"><IonIcon icon={home} /><IonLabel>Home</IonLabel></IonTabButton>
          <IonTabButton tab="evaluate" href="/evaluate"><IonIcon icon={playCircle} /><IonLabel>Evaluate</IonLabel></IonTabButton>
          <IonTabButton tab="results" href="/results"><IonIcon icon={barChart} /><IonLabel>Results</IonLabel></IonTabButton>
          <IonTabButton tab="settings" href="/settings"><IonIcon icon={settings} /><IonLabel>Settings</IonLabel></IonTabButton>
        </IonTabBar>
      </IonTabs>
    </IonReactRouter>
  </IonApp>
);

export default App;
