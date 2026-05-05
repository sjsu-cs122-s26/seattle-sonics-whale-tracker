"use client";

import { useRef, useState } from "react";

interface TopKItem {
  label: string;
  confidence: number;
}

interface Prediction {
  label: string;
  confidence: number;
  top_k?: TopKItem[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${apiUrl}/predict?top_k=5`, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API returned ${res.status}: ${text}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header>
        <h1>Fish Classifier</h1>
        <p>Upload a photo to identify the species</p>
      </header>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="upload-area" onClick={() => inputRef.current?.click()}>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
            <div className="upload-hint">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {file
                ? <strong>{file.name}</strong>
                : <><strong>Click to choose an image</strong><span>PNG, JPG, WEBP supported</span></>
              }
            </div>
          </div>

          {preview && <img src={preview} alt="Selected image preview" className="preview" />}

          <button className="btn" type="submit" disabled={!file || loading}>
            {loading ? "Classifying…" : "Classify"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <p className="result-label">{result.label}</p>
            <p className="result-confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </p>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
              />
            </div>

            {result.top_k && result.top_k.length > 1 && (
              <>
                <p className="top-k-title">Top predictions</p>
                <ul className="top-k">
                  {result.top_k.map((item) => (
                    <li key={item.label}>
                      <span>{item.label}</span>
                      <span>{(item.confidence * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
}
