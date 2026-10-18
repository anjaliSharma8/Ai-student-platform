import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import { execFile } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json());

// Helper to execute Python CLI safely
function runPython(args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    execFile('python3', ['cli_api.py', ...args], { maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python execution error:', stderr || error.message);
        return reject(new Error(stderr || error.message));
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (err: any) {
        console.error('JSON parse error from Python output:', stdout);
        reject(new Error(`Failed to parse Python JSON output: ${err.message}`));
      }
    });
  });
}

// API Routes
app.get('/api/data', async (_req: Request, res: Response) => {
  try {
    const data = await runPython(['get_data']);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/reset', async (_req: Request, res: Response) => {
  try {
    const data = await runPython(['reset_sample']);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/progress', async (req: Request, res: Response) => {
  try {
    const { taskId, pct } = req.body;
    if (!taskId || pct === undefined) {
      return res.status(400).json({ error: 'taskId and pct are required' });
    }
    const data = await runPython(['update_progress', String(taskId), String(pct)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/plan', async (_req: Request, res: Response) => {
  try {
    const data = await runPython(['run_planning']);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/astar', async (req: Request, res: Response) => {
  try {
    const horizon = req.body.horizon || 7;
    const data = await runPython(['run_astar', String(horizon)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/csp', async (req: Request, res: Response) => {
  try {
    const horizon = req.body.horizon || 7;
    const data = await runPython(['run_csp', String(horizon)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/bayesian', async (req: Request, res: Response) => {
  try {
    const evidence = req.body.evidence || {};
    const data = await runPython(['run_bayesian', JSON.stringify(evidence)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/benchmark', async (req: Request, res: Response) => {
  try {
    const horizon = req.body.horizon || 7;
    const data = await runPython(['run_benchmark', String(horizon)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/subject', async (req: Request, res: Response) => {
  try {
    const subject = req.body;
    const data = await runPython(['save_subject', JSON.stringify(subject)]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/subject/:id', async (req: Request, res: Response) => {
  try {
    const data = await runPython(['delete_subject', req.params.id]);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
