/**
 * @fileoverview Serwer Node.js do analizy CV za pomocą Gemini AI.
 * Obsługuje przesyłanie plików, pobieranie ogłoszeń o pracę
 * i generowanie analizy.
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import axios from 'axios';
import { createRequire } from 'module';
import rateLimit from 'express-rate-limit';
import nodemailer from 'nodemailer';
import validator from 'validator';
const require = createRequire(import.meta.url);
// pdf-parse (bez demo index)
const pdfParse = require('pdf-parse/lib/pdf-parse.js');
import mammoth from 'mammoth';

import { GoogleGenerativeAI } from "@google/generative-ai";
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

const app = express();
const PORT = process.env.PORT || 5000;

// ==== GEMINI ====
if (!process.env.GEMINI_API_KEY) {
  console.error("❌ Brak GEMINI_API_KEY w .env");
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// ==== LIMITY ====
const LIMITS = {
  cvChars: 20000,
  additionalDescriptionChars: 2000,
  jobSingleChars: 8000,
  jobTotalChars: 20000,
  summaryChars: 800,
  itemChars: 200,
  maxOutputTokens: 1024
};
const clampText = (str, max) => {
  const s = String(str || '');
  return s.length > max ? s.slice(0, max) + '…' : s;
};

// Rate limit: max 10 zapytań / 15 minut z 1 IP
const contactLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false
});

// Transport e-mail (SMTP z .env)
const mailTransport = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: Number(process.env.SMTP_PORT || 587),
  secure: false,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS
  }
});

// ==== UPLOAD ====
if (!fs.existsSync('uploads')) fs.mkdirSync('uploads', { recursive: true });
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.pdf', '.docx'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) cb(null, true);
    else cb(new Error('Dozwolone tylko pliki PDF i DOCX'));
  }
});

// ==== MIDDLEWARE ====
const allowedOrigins = [
  'https://cv-analyzer-frontend.vercel.app',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  'http://localhost:3000',
  'http://localhost:5000'
];
app.use(cors({ origin: allowedOrigins, credentials: true }));

// AB middleware
app.use((req, res, next) => {
  req.abVariant = (req.headers['x-ab-variant'] === 'B') ? 'B' : 'A';
  next();
});


const jsonParser = express.json({ limit: '1mb' });

// ==== HEALTH ====
app.get('/api/health', (req, res) => {
  res.json({ ok: true, hasGeminiKey: !!process.env.GEMINI_API_KEY });
});

// ==== HELPERS (Twoje + nowe) ====
const parseMaybeJSON = (value) => {
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return value; }
};
const normalizeToArray = (value) => {
  const v = parseMaybeJSON(value);
  if (Array.isArray(v)) return v;
  if (!v) return [];
  if (typeof v === 'string') {
    return v.split(/\r?\n|,|;/).map(s => s.trim()).filter(Boolean);
  }
  return [];
};

const extractAndParseJSON = (input) => {
  if (!input) throw new Error('Pusta odpowiedź modelu');
  let t = String(input).replace(/^\uFEFF/, '').trim();
  if (t.startsWith('```')) {
    t = t.replace(/^```(?:json)?\s*/i, '');
    t = t.replace(/```$/i, '');
  }
  t = t.trim();

  const cleanAndParse = (candidate) => {
    candidate = candidate.replace(/[“”„”]/g, '"').replace(/[‘’]/g, "'");
    candidate = candidate.replace(/,\s*(?=[}```])/g, '');
    candidate = candidate.replace(/\/\/[^\n\r]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    return JSON.parse(candidate);
  };

  try { return cleanAndParse(t); } catch {}

  const findBalancedJSON = (s) => {
    const openers = ['{', '['];
    const closers = { '{': '}', '[': ']' };
    let firstIdx = -1;
    let opener = null;
    for (let i = 0; i < s.length; i++) {
      if (openers.includes(s[i])) { firstIdx = i; opener = s[i]; break; }
    }
    if (firstIdx === -1) return null;
    const closer = closers[opener];
    let depth = 0, inStr = false, escape = false;
    for (let i = firstIdx; i < s.length; i++) {
      const ch = s[i];
      if (inStr) {
        if (escape) escape = false;
        else if (ch === '\\') escape = true;
        else if (ch === '"') inStr = false;
        continue;
      } else {
        if (ch === '"') { inStr = true; continue; }
        if (ch === opener) depth++;
        if (ch === closer) depth--;
        if (depth === 0) return s.slice(firstIdx, i + 1);
      }
    }
    return null;
  };

  const candidate = findBalancedJSON(t);
  if (candidate) {
    try { return cleanAndParse(candidate); } catch {}
  }

  const startObj = t.indexOf('{');
  const endObj = t.lastIndexOf('}');
  if (startObj !== -1 && endObj !== -1 && endObj > startObj) {
    const slice = t.slice(startObj, endObj + 1);
    try { return cleanAndParse(slice); } catch {}
  }
  const startArr = t.indexOf('[');
  const endArr = t.lastIndexOf(']');
  if (startArr !== -1 && endArr !== -1 && endArr > startArr) {
    const slice = t.slice(startArr, endArr + 1);
    try { return cleanAndParse(slice); } catch {}
  }

  throw new Error('Nie udało się sparsować JSON z odpowiedzi modelu');
};

const ensureAnalysisShape = (a = {}) => ({
  podsumowanie: a.podsumowanie || '',
  dopasowanie: {
    mocne_strony: Array.isArray(a?.dopasowanie?.mocne_strony) ? a.dopasowanie.mocne_strony : [],
    obszary_do_poprawy: Array.isArray(a?.dopasowanie?.obszary_do_poprawy) ? a.dopasowanie.obszary_do_poprawy : []
  },
  pytania: {
    kompetencje_miekkie: Array.isArray(a?.pytania?.kompetencje_miekkie) ? a.pytania.kompetencje_miekkie : [],
    kompetencje_twarde: Array.isArray(a?.pytania?.kompetencje_twarde) ? a.pytania.kompetencje_twarde : []
  }
});

// --- STAR: kształt odpowiedzi i sanitizacja ---
const ensureStarShape = (a = {}) => ({
  answer: String(a.answer || '').trim(),
  feedback: Array.isArray(a.feedback) ? a.feedback : [],
  grade: String(a.grade || '').trim(), // np. A, B, C (albo 1-6)
  scores: {
    clarity: Number(a?.scores?.clarity ?? 0),     // 0-100
    impact: Number(a?.scores?.impact ?? 0),
    metrics: Number(a?.scores?.metrics ?? 0),
    ownership: Number(a?.scores?.ownership ?? 0),
    relevance: Number(a?.scores?.relevance ?? 0),
  },
  tips: Array.isArray(a.tips) ? a.tips : [],
  red_flags: Array.isArray(a.red_flags) ? a.red_flags : [],
  length_chars: Number(a.length_chars ?? 0),
});

const STAR_LIMITS = {
  maxFieldChars: 1200,      // każdy z S/T/A/R
  maxOutputChars: 2000,     // gotowa odpowiedź
  maxItemChars: 200
};

const hasNumbers = (txt='') => /\d/.test(txt);
const sanitizeStringArray = (arr, maxLen) => {
  if (!Array.isArray(arr)) return [];
  return arr.map(x => clampText(String(x || '').trim(), maxLen)).filter(Boolean);
};

// Pobranie ogłoszenia (Readability)
const getJobDescriptionWithReadability = async (url) => {
  try {
    const response = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
      timeout: 10000
    });
    const doc = new JSDOM(response.data, { url });
    const reader = new Readability(doc.window.document);
    const article = reader.parse();
    const textContent = article
      ? article.textContent.replace(/\s+/g, ' ').trim()
      : 'Nie udało się pobrać treści ogłoszenia.';
    return clampText(textContent, LIMITS.jobSingleChars);
  } catch (error) {
    console.error("❌ Błąd pobierania ogłoszenia:", error.message);
    return 'Nie udało się pobrać treści ogłoszenia.';
  }
};

// Odczyt CV
const extractTextFromCV = async (filePath) => {
  try {
    const ext = path.extname(filePath).toLowerCase();
    if (!fs.existsSync(filePath)) throw new Error('Plik nie istnieje');

    if (ext === '.pdf') {
      const data = await fs.promises.readFile(filePath);
      const { text } = await pdfParse(data);
      return clampText((text || '').trim(), LIMITS.cvChars);
    } else if (ext === '.docx') {
      const result = await mammoth.extractRawText({ path: filePath });
      const textContent = (result.value || '').trim();
      return clampText(textContent, LIMITS.cvChars);
    } else {
      throw new Error('Obsługiwane są tylko pliki PDF i DOCX.');
    }
  } catch (error) {
    console.error('Błąd odczytu pliku:', error);
    throw error;
  }
};

// Pobranie tekstu z odpowiedzi Gemini (inlineData/plain)
const getLLMOutputString = async (resp) => {
  try {
    const parts = resp?.candidates?.[0]?.content?.parts || [];
    const jsonInline = parts.find(p => p?.inlineData && /json/i.test(p.inlineData.mimeType || ''));
    if (jsonInline?.inlineData?.data) {
      return Buffer.from(jsonInline.inlineData.data, 'base64').toString('utf-8');
    }
  } catch {}
  try { return await resp.text(); } catch { return ''; }
};

// Wywołanie Gemini z fallbackiem
const generateWithFallback = async (prompt) => {
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      generationConfig: { temperature: 0.2, maxOutputTokens: LIMITS.maxOutputTokens }
    });
    const result = await model.generateContent(prompt);
    return await getLLMOutputString(result.response);
  } catch (e) {
    const msg = String(e?.message || e);
    if (msg.includes('valid JSON') || msg.toLowerCase().includes('json')) {
      console.warn('Gemini JSON error - retry text/plain:', msg);
      const modelPlain = genAI.getGenerativeModel({
        model: "gemini-1.5-flash",
        generationConfig: { responseMimeType: "text/plain", temperature: 0.2, maxOutputTokens: LIMITS.maxOutputTokens }
      });
      const retry = await modelPlain.generateContent(prompt);
      try { return await retry.response.text(); }
      catch {
        const parts = retry?.response?.candidates?.[0]?.content?.parts || [];
        return parts.map(p => p?.text || '').join('') || '';
      }
    }
    throw e;
  }
};

// ====== NOWE: NLP pomocnicze ======
const STOPWORDS_PL = new Set(['i','oraz','lub','albo','w','na','do','z','za','o','u','jak','że','to','jest','są','być','przy','dla','od','po','pod','nad','bez','też','np','itp','itd']);
const simpleKeywords = (text = '', { minLen = 3, topN = 80 } = {}) => {
  const freq = new Map();
  const words = (text || '')
    .toLowerCase()
    .replace(/[^a-ząćęłńóśźż0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length >= minLen && !STOPWORDS_PL.has(w));
  for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);
  return [...freq.entries()].sort((a,b) => b[1]-a[1]).slice(0, topN).map(([w]) => w);
};
const jaccard = (arrA = [], arrB = []) => {
  const A = new Set(arrA), B = new Set(arrB);
  const inter = [...A].filter(x => B.has(x)).length;
  const uni = new Set([...A, ...B]).size || 1;
  return inter / uni; // 0..1
};

// --- JD DIFF: pomocnicze ---
const DIFF_LIMITS = { maxJobs: 5, maxReqItemChars: 180 };

const normalizeWhitespace = (s='') => String(s).replace(/\s+/g, ' ').trim();

const dedupKeepOrder = (arr=[]) => {
  const seen = new Set(); const out = [];
  for (const x of arr) { const y = x.trim(); if (!y || seen.has(y)) continue; seen.add(y); out.push(y); }
  return out;
};

const intersectMany = (arrays) => {
  if (!arrays.length) return [];
  return arrays.reduce((acc, cur) => acc.filter(x => cur.includes(x)));
};

const llmExtractRequirements = async (text) => {
  const prompt = `
Wypisz wymagania/kompetencje z poniższego ogłoszenia o pracę jako czystą listę JSON (tablica stringów, bez markdown). Każdy punkt krótki (<= ${DIFF_LIMITS.maxReqItemChars} znaków).
OGŁOSZENIE:
${String(text).slice(0, 8000)}
ZWRÓĆ TYLKO TABLICĘ JSON.`;
  const raw = await generateWithFallback(prompt);
  try {
    const arr = extractAndParseJSON(raw);
    return sanitizeStringArray(arr, DIFF_LIMITS.maxReqItemChars);
  } catch {
    // fallback: proste wyłuskanie "bulletów"
    const bullets = text.split(/\n|•|- |\* /).map(x => x.trim()).filter(x => x.length > 5);
    return sanitizeStringArray(bullets.slice(0, 30), DIFF_LIMITS.maxReqItemChars);
  }
};

const fitScoreFromKeywords = (cvText, reqs) => {
  const cvKW = new Set(simpleKeywords(cvText, { topN: 120 }));
  const reqKW = new Set(simpleKeywords(reqs.join(' '), { topN: 120 }));
  const overlap = [...reqKW].filter(k => cvKW.has(k)).length;
  const denom = Math.max(1, reqKW.size);
  return Math.round((overlap / denom) * 100); // 0..100
};


// Klasyfikator branża/rola (Gemini)
const classifyText = async (text, kind = 'CV') => {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const prompt = `
Jesteś klasyfikatorem. Dla danego tekstu (${kind}) zwróć JSON:
{"industry":"IT/Software|Finanse/Księgowość|Bankowość/Ubezpieczenia|Analityka/BI|Marketing/Digital|Sprzedaż/CS|HR/Rekrutacja|Logistyka/Operacje|Produkcja/Inżynieria|PM/PMO|Inne","role_hint":"krótka rola/stack","seniority":"Junior|Mid|Senior|Unknown"}
Tekst:
${String(text || '').slice(0, 4000)}
Zwróć tylko JSON.`;
    const res = await model.generateContent(prompt);
    const raw = await res.response.text();
    const cleaned = raw.replace(/```json|```/g, '').trim();
    return JSON.parse(cleaned);
  } catch {
    return { industry: "Inne", role_hint: "", seniority: "Unknown" };
  }
};

// ====== ENDPOINTY ======

// 1) ANALIZA CV (bez JD) – POST /api/analyze-cv
app.post('/api/analyze-cv', upload.single('cv'), async (req, res) => {
  let uploadedFilePath;
  try {
    if (!req.file) return res.status(400).json({ error: 'Brak pliku CV' });
    uploadedFilePath = req.file.path;
    let cvText = await extractTextFromCV(uploadedFilePath);

    const additionalDescription = typeof req.body?.additionalDescription === 'string'
      ? clampText(req.body.additionalDescription, LIMITS.additionalDescriptionChars) : '';

    if (additionalDescription) {
      cvText += `\n\n[DODATKOWY OPIS]\n${additionalDescription}`;
    }

    const cvClass = await classifyText(cvText, 'CV');

    // Krótka analiza CV
    const prompt = `
Zanalizuj CV kandydata i zwróć JSON:
{
 "podsumowanie":"2-3 zdania",
 "dopasowanie":{"mocne_strony":["..."],"obszary_do_poprawy":["..."]},
 "rekomendacje_cv":["konkret #1","konkret #2","konkret #3"]
}
CV:
${cvText}`;
    const llmRaw = await generateWithFallback(prompt);
    let analysis = ensureAnalysisShape({});
    try {
      const parsed = extractAndParseJSON(llmRaw);
      analysis = ensureAnalysisShape(parsed);
      // --- enforce plan limits on number of questions ---
analysis.pytania.kompetencje_miekkie =
  sanitizeStringArray(analysis.pytania.kompetencje_miekkie, LIMITS.itemChars).slice(0, numSoft);

analysis.pytania.kompetencje_twarde =
  sanitizeStringArray(analysis.pytania.kompetencje_twarde, LIMITS.itemChars).slice(0, numHard);
    } catch {
      analysis.podsumowanie = 'Nie udało się poprawnie sparsować odpowiedzi modelu.';
      analysis.rawResponse = String(llmRaw || '').slice(0, 2000);
    }

    // Proste % (bez JD)
    const total = analysis.dopasowanie.mocne_strony.length + analysis.dopasowanie.obszary_do_poprawy.length;
    analysis.dopasowanie_procentowe = total > 0
      ? Math.round((analysis.dopasowanie.mocne_strony.length / total) * 100)
      : 0;

    analysis.meta = { industry_cv: cvClass.industry, role_cv: cvClass.role_hint };
    analysis.warning_mismatch = false;

    return res.json({ status: 'success', analysis });
  } catch (error) {
    console.error("🔥 /api/analyze-cv", error);
    return res.status(500).json({ error: 'Błąd analizy CV', details: error.message });
  } finally {
    if (uploadedFilePath) fs.unlink(uploadedFilePath, () => {});
  }
});

// 2) ANALIZA JD (bez CV) – POST /api/analyze-jd
app.post('/api/analyze-jd', jsonParser, async (req, res) => {
  try {
    const urls = normalizeToArray(req.body?.jobUrls || req.body?.urls);
    if (!urls.length) return res.status(400).json({ error: 'Brak linków do ogłoszeń' });

    const jobDescriptions = await Promise.all(urls.map(getJobDescriptionWithReadability));
    const combinedJD = clampText(jobDescriptions.join("\n\n---\n\n"), LIMITS.jobTotalChars);

    const jdClass = await classifyText(combinedJD, 'JD');

    const prompt = `
Streść ogłoszenia o pracę i zwróć JSON:
{"podsumowanie":"1-2 zdania","stack":["tech/obszary"],"kluczowe_wymagania":["..."],"nice_to_have":["..."]}
OGŁOSZENIA:
${combinedJD}`;
    const llmRaw = await generateWithFallback(prompt);
    let out;
    try { out = extractAndParseJSON(llmRaw); }
    catch { out = { podsumowanie: 'Brak parsowalnej odpowiedzi', raw: String(llmRaw).slice(0,1500) }; }

    return res.json({ status: 'success', jd: out, meta: { industry_jd: jdClass.industry, role_jd: jdClass.role_hint } });
  } catch (error) {
    console.error("🔥 /api/analyze-jd", error);
    return res.status(500).json({ error: 'Błąd analizy JD', details: error.message });
  }
});

// 3) GENERACJA PYTAŃ – POST /api/generate-questions
app.post('/api/generate-questions', jsonParser, async (req, res) => {
  const normalizeIndustry = (s='') => s.replace(/\s*\/\s*/g, '/').trim();
  const selectedIndustryRaw = typeof req.body?.selectedIndustry === 'string' ? req.body.selectedIndustry : '';
  const selectedIndustry = normalizeIndustry(selectedIndustryRaw);

  try {
    const { plan = 'free' } = req.body || {}; // Użyj destrukturyzacji tylko dla 'plan'
    const urls = normalizeToArray(req.body?.jobUrls || req.body?.urls);

    const isFree = String(plan).toLowerCase() === 'free';
    const numSoft = isFree ? 2 : 7;
    const numHard = isFree ? 2 : 10;

    let combinedJD = '';
    if (urls.length) {
      const jobDescriptions = await Promise.all(urls.map(getJobDescriptionWithReadability));
      combinedJD = clampText(jobDescriptions.join("\n\n---\n\n"), LIMITS.jobTotalChars);
    }

    const jdClass = await classifyText(combinedJD || selectedIndustry || 'Inne', 'JD');

    const prompt = `
Na bazie branży/ogłoszenia wygeneruj pytania (JSON):
{"pytania":{"kompetencje_miekkie":[${Array(numSoft).fill('"..."').join(',')}],"kompetencje_twarde":[${Array(numHard).fill('"..."').join(',')}]}}

BRANŻA (UI): ${selectedIndustry || 'Nie podano'}
KLASYFIKACJA JD: industry="${jdClass.industry}", role="${jdClass.role_hint}"
OGŁOSZENIA:
${combinedJD || '(brak)'}
`;
    const llmRaw = await generateWithFallback(prompt);
    let out = { pytania: { kompetencje_miekkie: [], kompetencje_twarde: [] } };
    try { out = extractAndParseJSON(llmRaw); } catch {}
    out.pytania = out.pytania || {};
    out.pytania.kompetencje_miekkie = sanitizeStringArray(out.pytania.kompetencje_miekkie, LIMITS.itemChars).slice(0, numSoft);
    out.pytania.kompetencje_twarde = sanitizeStringArray(out.pytania.kompetencje_twarde, LIMITS.itemChars).slice(0, numHard);

    return res.json({ status: 'success', ...out, meta: { industry_jd: jdClass.industry, role_jd: jdClass.role_hint } });
  } catch (error) {
    console.error("🔥 /api/generate-questions", error);
    return res.status(500).json({ error: 'Błąd generacji pytań', details: error.message });
  }
});

// 4) COACH STAR – POST /api/star-coach
app.post('/api/star-coach', jsonParser, async (req, res) => {
  try {
    const { situation='', task='', action='', result='', role='', language='pl', tone='concise', plan='free' } = req.body || {};

    // Twarde limity wejścia
    const S = clampText(String(situation), STAR_LIMITS.maxFieldChars);
    const T = clampText(String(task), STAR_LIMITS.maxFieldChars);
    const A = clampText(String(action), STAR_LIMITS.maxFieldChars);
    const R = clampText(String(result), STAR_LIMITS.maxFieldChars);

    // Heurystyka: czy są liczby/metryki?
    const metricsHint = [R, A].some(hasNumbers);

    // Prompt z twardym JSON-em
    const lang = (language || 'pl').toLowerCase().startsWith('en') ? 'en' : 'pl';
    const toneMap = { concise: (lang==='pl'?'zwięźle':'concise'), professional: (lang==='pl'?'profesjonalnie':'professional'), enthusiastic: (lang==='pl'?'energicznie':'enthusiastic') };
    const toneLabel = toneMap[tone] || toneMap.concise;

    const prompt = `
Jesteś coachem odpowiedzi metodą STAR. Z poniższych pól (Sytuacja, Zadanie, Akcja, Rezultat) zbuduj krótką, mocną odpowiedź i oceń ją.

JEZYK ODPOWIEDZI: ${lang==='pl'?'polski':'english'}
STYL: ${toneLabel}
KONTEKST ROLI (opcjonalny): ${role || (lang==='pl'?'(brak)':'(none)')}

WYTYCZNE:
- Odpowiedź ma mieścić się w ${STAR_LIMITS.maxOutputChars} znakach (krótko, esencja).
- Używaj liczb/KPI jeżeli istnieją w wejściu; jeśli brak, zasugeruj jak je dodać.
- Zachowaj strukturę STAR (1 akapit, logiczne zdania).
- Unikaj żargonu i zbyt długich wstępów.
- Ton: ${toneLabel}.
- Zwróć TYLKO JSON bez markdown.

FORMAT JSON:
{
  "answer": "gotowa odpowiedź STAR (max ${STAR_LIMITS.maxOutputChars} znaków)",
  "feedback": ["1-3 krótkie uwagi konstruktywne (max ${STAR_LIMITS.maxItemChars} znaków każda)"],
  "grade": "A|B|C|D|E",
  "scores": {
    "clarity": 0-100,
    "impact": 0-100,
    "metrics": 0-100,
    "ownership": 0-100,
    "relevance": 0-100
  },
  "tips": ["1-3 konkretne wskazówki jak podnieść ocenę (max ${STAR_LIMITS.maxItemChars} znaków)"],
  "red_flags": ["opcjonalne ryzyka/braki jeśli są (max ${STAR_LIMITS.maxItemChars} znaków)"],
  "length_chars": <długość pola answer>
}

DANE WEJŚCIOWE:
[S] ${S}
[T] ${T}
[A] ${A}
[R] ${R}

WSKAZÓWKA: ${metricsHint ? (lang==='pl'?'Wejście zawiera liczby — włącz je do odpowiedzi.':'Input already includes metrics — use them.') : (lang==='pl'?'Wejście nie zawiera liczb — zasugeruj mierniki.':'No metrics detected — suggest KPIs.')}
`.trim();

    const llmRaw = await generateWithFallback(prompt);
    let star = ensureStarShape({});
    try {
      const parsed = extractAndParseJSON(llmRaw);
      star = ensureStarShape(parsed);
    } catch (e) {
      // Fallback najprostszy, gdyby JSON nie przeszedł
      const fallbackAnswer = `${lang==='pl'
        ? 'S: ' + S + ' | T: ' + T + ' | A: ' + A + ' | R: ' + R
        : 'S: ' + S + ' | T: ' + T + ' | A: ' + A + ' | R: ' + R}`;
      star.answer = clampText(fallbackAnswer, STAR_LIMITS.maxOutputChars);
      star.feedback = [(lang==='pl'?'Nie udało się w pełni przetworzyć odpowiedzi AI.':'Could not fully parse AI output.')];
      star.grade = 'C';
      star.scores = { clarity: 50, impact: 45, metrics: metricsHint?60:30, ownership: 50, relevance: 60 };
      star.tips = [(lang==='pl'?'Dodaj liczby/rezultaty (%, PLN, czas).':'Add metrics (%, $, time).')];
      star.red_flags = [];
      star.length_chars = star.answer.length;
    }

    // Sanitizacja i twarde limity
    star.answer = clampText(star.answer, STAR_LIMITS.maxOutputChars);
    star.feedback = sanitizeStringArray(star.feedback, STAR_LIMITS.maxItemChars).slice(0, 3);
    star.tips = sanitizeStringArray(star.tips, STAR_LIMITS.maxItemChars).slice(0, 3);
    star.red_flags = sanitizeStringArray(star.red_flags, STAR_LIMITS.maxItemChars).slice(0, 3);
    star.length_chars = star.answer.length;

    // Analityka koszykowa (lekka) w logu
    console.log('[STAR]', {
      ab: req.abVariant,
      grade: star.grade,
      metrics: star.scores.metrics,
      length: star.length_chars
    });

    return res.json({ status: 'success', star });
  } catch (error) {
    console.error('🔥 /api/star-coach', error);
    return res.status(500).json({ error: 'Błąd generowania odpowiedzi STAR', details: error.message });
  }
});

// 5) JD DIFF — POST /api/jd-diff
// Form-data (z CV): cv(file), jobUrls(json lub newline string)
// albo JSON bez pliku: { jobUrls: [...], cvText?: "..." }
app.post('/api/jd-diff', upload.single('cv'), async (req, res) => {
  let uploadedFilePath;
  try {
    const isMultipart = !!req.file || (req.headers['content-type']||'').includes('multipart/form-data');
    const urlsRaw = isMultipart ? (req.body?.jobUrls ?? req.body?.urls) : (req.body?.jobUrls ?? req.body?.urls);
    let urls = normalizeToArray(urlsRaw);
    if (!urls.length) return res.status(400).json({ error: 'Podaj 2–5 linków do ogłoszeń (jobUrls).' });
    if (urls.length < 2 || urls.length > DIFF_LIMITS.maxJobs) {
      return res.status(400).json({ error: `Dozwolone 2–${DIFF_LIMITS.maxJobs} ogłoszeń.` });
    }

    // CV tekst: z pliku, albo z JSON (cvText), albo puste
    let cvText = '';
    if (req.file) {
      uploadedFilePath = req.file.path;
      cvText = await extractTextFromCV(uploadedFilePath);
    } else if (!isMultipart && typeof req.body?.cvText === 'string') {
      cvText = clampText(req.body.cvText, LIMITS.cvChars);
    }

    // Pobierz i przygotuj JD
    const jdTexts = await Promise.all(urls.map(getJobDescriptionWithReadability));
    const items = [];
    for (let i = 0; i < jdTexts.length; i++) {
      const url = urls[i];
      const text = normalizeWhitespace(jdTexts[i] || '');
      const reqs = dedupKeepOrder(await llmExtractRequirements(text));
      items.push({ url, text, requirements: reqs });
    }

    // Wspólne i unikalne wymagania
    const reqArrays = items.map(x => x.requirements);
    const common = intersectMany(reqArrays);
    const uniqueBy = {};
    items.forEach((it, idx) => {
      const others = dedupKeepOrder([].concat(...reqArrays.filter((_, j) => j !== idx)));
      uniqueBy[it.url] = it.requirements.filter(r => !others.includes(r));
    });

    // Dopasowanie CV → każdy JD
    const perJob = items.map(it => {
      const score = cvText ? fitScoreFromKeywords(cvText, it.requirements) : null;
      return {
        url: it.url,
        total_requirements: it.requirements.length,
        fit_score: score, // null gdy brak CV
        top_missing: cvText
          ? it.requirements
              .filter(r => fitScoreFromKeywords(r, [r]) === 0) // prosta heurystyka: brak słów kluczowych z punktu
              .slice(0, 5)
          : []
      };
    });

    // Syntetyczne słowa kluczowe dla wglądu (nieobowiązkowe)
    const jdKW = simpleKeywords(items.map(i => i.text).join(' '), { topN: 60 });

    // Meta
    const meta = {
      jobs_count: items.length,
      with_cv: !!cvText,
      ab_variant: req.abVariant
    };

    // Log
    console.log('[JD-DIFF]', {
      jobs: items.length,
      with_cv: !!cvText,
      common_count: common.length
    });

    return res.json({
  status: 'success',
  diff: {
    common_requirements: common,
    unique_by_job: uniqueBy,
    per_job: perJob,
    jd_keywords: jdKW,
    // NOWE: pełne wymagania per ogłoszenie (przydadzą się do CSV w froncie)
    requirements_by_job: items.map((it, idx) => ({
      id: `job_${idx+1}`,
      label: it.url,
      requirements: it.requirements
    }))
  },
  meta
});
  } catch (err) {
    console.error('🔥 /api/jd-diff', err);
    return res.status(500).json({ error: 'Błąd porównania ogłoszeń', details: err.message });
  } finally {
    if (uploadedFilePath) fs.unlink(uploadedFilePath, () => {});
  }
});

// 7) JD DIFF (from text) — POST /api/jd-diff-from-text
// JSON body: { jdTexts: [string, ...], labels?: [string,...], cvText?: string }
app.post('/api/jd-diff-from-text', jsonParser, async (req, res) => {
  try {
    const jdTexts = Array.isArray(req.body?.jdTexts) ? req.body.jdTexts.map(x => normalizeWhitespace(String(x||''))) : [];
    const labels = Array.isArray(req.body?.labels) ? req.body.labels.map(x => String(x||'')) : [];
    const cvText = typeof req.body?.cvText === 'string' ? clampText(req.body.cvText, LIMITS.cvChars) : '';

    if (jdTexts.length < 2 || jdTexts.length > DIFF_LIMITS.maxJobs) {
      return res.status(400).json({ error: `Podaj 2–${DIFF_LIMITS.maxJobs} treści ogłoszeń w polu jdTexts.` });
    }

    // Ekstrakcja wymagań z każdego JD
    const items = [];
    for (let i = 0; i < jdTexts.length; i++) {
      const text = jdTexts[i];
      const reqs = dedupKeepOrder(await llmExtractRequirements(text));
      const label = labels[i] && labels[i].trim() ? labels[i].trim() : `JD #${i+1}`;
      items.push({ label, text, requirements: reqs });
    }

    const reqArrays = items.map(x => x.requirements);
    const common = intersectMany(reqArrays);

    const uniqueBy = {};
    items.forEach((it, idx) => {
      const others = dedupKeepOrder([].concat(...reqArrays.filter((_, j) => j !== idx)));
      uniqueBy[it.label] = it.requirements.filter(r => !others.includes(r));
    });

    const perJob = items.map(it => {
      const score = cvText ? fitScoreFromKeywords(cvText, it.requirements) : null;
      return {
        label: it.label,
        total_requirements: it.requirements.length,
        fit_score: score,
        top_missing: cvText
          ? it.requirements
              .filter(r => fitScoreFromKeywords(r, [r]) === 0)
              .slice(0, 5)
          : []
      };
    });

    const jdKW = simpleKeywords(items.map(i => i.text).join(' '), { topN: 60 });
    const meta = { jobs_count: items.length, with_cv: !!cvText, ab_variant: req.abVariant };

    console.log('[JD-DIFF-TEXT]', { jobs: items.length, with_cv: !!cvText, common_count: common.length });

    return res.json({
      status: 'success',
      diff: {
        common_requirements: common,
        unique_by_job: uniqueBy,
        per_job: perJob,
        jd_keywords: jdKW,
        // pełne wymagania per JD (z etykietami)
        requirements_by_job: items.map((it, idx) => ({
          id: `job_${idx+1}`,
          label: it.label,
          requirements: it.requirements
        }))
      },
      meta
    });
  } catch (err) {
    console.error('🔥 /api/jd-diff-from-text', err);
    return res.status(500).json({ error: 'Błąd porównania ogłoszeń (tekst)', details: err.message });
  }
});

// 6) SCALONY FLOW – POST /api/analyze-cv-multiple
app.post('/api/analyze-cv-multiple', upload.single('cv'), async (req, res) => {
  console.log("---- /api/analyze-cv-multiple ----");
  console.log("Plik:", req.file?.originalname);
  console.log("Body:", req.body);

  let uploadedFilePath;
  try {
    if (!req.file) return res.status(400).json({ error: 'Brak przesłanego pliku CV' });
    if (!process.env.GEMINI_API_KEY) {
      return res.status(400).json({ error: 'Brak GEMINI_API_KEY w .env' });
    }

    uploadedFilePath = req.file.path;

    // Wejście
    const urls = normalizeToArray(req.body?.jobUrls || req.body?.urls);
    const additionalDescription = typeof req.body?.additionalDescription === 'string'
      ? clampText(req.body.additionalDescription, LIMITS.additionalDescriptionChars) : '';
    const plan = typeof req.body?.plan === 'string' ? req.body.plan : 'free';
    const selectedIndustry = typeof req.body?.selectedIndustry === 'string' ? req.body.selectedIndustry : '';

    // CV
    let cvText = await extractTextFromCV(uploadedFilePath);
    if (additionalDescription) cvText += `\n\n[DODATKOWY OPIS KANDYDATA]\n${additionalDescription}`;

    // JD
    const jobDescriptions = urls.length ? await Promise.all(urls.map(getJobDescriptionWithReadability)) : [];
    const combinedJD = clampText(
      jobDescriptions.length ? jobDescriptions.join("\n\n---\n\n") : 'Brak ogłoszeń lub nie udało się pobrać treści.',
      LIMITS.jobTotalChars
    );

    // Klasyfikacja
    const cvClass = await classifyText(cvText, 'CV');
    const jdClass = await classifyText(combinedJD, 'JD');

    // Efektywna branża: priorytet JD -> potem UI -> CV
    const effectiveIndustry =
      (jdClass.industry && jdClass.industry !== 'Inne') ? jdClass.industry :
      (selectedIndustry && selectedIndustry.trim())     ? selectedIndustry.trim() :
      (cvClass.industry || 'Inne');

    // Słowa kluczowe + overlap
    const cvKW = simpleKeywords(cvText, { topN: 80 });
    const jdKW = simpleKeywords(combinedJD, { topN: 80 });
    const kwOverlapPct = Math.round(jaccard(cvKW, jdKW) * 100);

    const isFree = String(plan).toLowerCase() === "free";
    const numSoft = isFree ? 2 : 7;
    const numHard = isFree ? 2 : 10;

    // Twarde reguły w prompcie
    const prompt = `
Jesteś ekspertem HR/Tech Recruiterem.

=== KLASA BRANŻOWA ===
CV: industry="${cvClass.industry}", role="${cvClass.role_hint}", seniority="${cvClass.seniority}"
JD: industry="${jdClass.industry}", role="${jdClass.role_hint}", seniority="${jdClass.seniority}"
UI selectedIndustry="${selectedIndustry || 'Nie podano'}"
Efektywna branża analizy: "${effectiveIndustry}"

=== REGUŁY ===
1) Jeśli branża CV i JD się różnią → PRIORYTET ma JD i efektywna branża.
2) Pytania generuj **pod JD/efektywną branżę**, nie pod CV przy konflikcie.
3) Rekomendacje CV mają zwiększać dopasowanie do JD (transferowalne kompetencje).
4) Wskaż ryzyko niedopasowania (CV≠JD) + zaproponuj most (kursy, projekty).

=== OGRANICZENIA FORMATU ===
- ZWRÓĆ WYŁĄCZNIE JSON bez komentarzy i bez markdown.
- Dokładnie ${numSoft} pozycji w "kompetencje_miekkie".
- Dokładnie ${numHard} pozycji w "kompetencje_twarde".
- Każda pozycja ma maksymalnie ${LIMITS.itemChars} znaków.

=== FORMAT JSON ===
{
  "podsumowanie": "2-4 zdania",
  "dopasowanie": {
    "mocne_strony": ["..."],
    "obszary_do_poprawy": ["..."],
    "ryzyko_niedopasowania": "..."
  },
  "pytania": {
    "kompetencje_miekkie": [${Array(numSoft).fill('"..."').join(',')}],
    "kompetencje_twarde": [${Array(numHard).fill('"..."').join(',')}]
  },
  "rekomendacje_cv": ["konkret #1","konkret #2","konkret #3"]
}

=== CV ===
${cvText}

=== OGŁOSZENIA (JD) ===
${combinedJD}
`;
    const llmRaw = await generateWithFallback(prompt);
    let analysis;
    try {
      const parsed = extractAndParseJSON(llmRaw);
      analysis = ensureAnalysisShape(parsed);
    } catch (parseError) {
      console.error("Błąd parsowania odpowiedzi z Gemini:", parseError);
      analysis = ensureAnalysisShape({});
      analysis.podsumowanie = 'Nie udało się poprawnie sparsować odpowiedzi modelu. Surowa odpowiedź (skrót) w polu rawResponse.';
      analysis.rawResponse = String(llmRaw || '').slice(0, 2000);
    }

    // Heurystyczny % dopasowania z capem wg overlapu
    const total = analysis.dopasowanie.mocne_strony.length + analysis.dopasowanie.obszary_do_poprawy.length;
    let basePct = total > 0 ? Math.round((analysis.dopasowanie.mocne_strony.length / total) * 100) : 0;
    if (kwOverlapPct < 10) basePct = Math.min(basePct, 35);
    else if (kwOverlapPct < 20) basePct = Math.min(basePct, 50);
    analysis.dopasowanie_procentowe = basePct;

    // Flaga mismatch + meta do debugowania
    const mismatch = (cvClass.industry !== 'Inne' && jdClass.industry !== 'Inne' && cvClass.industry !== jdClass.industry);
    analysis.warning_mismatch = !!mismatch;
    analysis.meta = {
      ...analysis.meta, // Zapewnij, że inne pola meta zostaną zachowane
      ab_variant: req.abVariant,
      industry_cv: cvClass.industry,
      industry_jd: jdClass.industry,
      industry_effective: effectiveIndustry,
      role_cv: cvClass.role_hint,
      role_jd: jdClass.role_hint,
      kw_overlap_pct: kwOverlapPct
    };

    // Logowanie metryk
    console.log('[AB]', {
      ab: req.abVariant,
      kw_overlap_pct: analysis?.meta?.kw_overlap_pct,
      industry_cv: analysis?.meta?.industry_cv,
      industry_jd: analysis?.meta?.industry_jd,
      warning_mismatch: analysis?.warning_mismatch,
      dopasowanie_pct: analysis?.dopasowanie_procentowe
    });

    return res.json({ status: 'success', analysis });
  } catch (error) {
    console.error("🔥 /api/analyze-cv-multiple", error);
    return res.status(500).json({ error: 'Błąd podczas analizy CV', details: error.message });
  } finally {
    if (uploadedFilePath) fs.unlink(uploadedFilePath, () => {});
  }
});

// --- AUTODETEKCJA BRANŻY (zostawiamy) ---
const INDUSTRY_KEYWORDS = {
  'IT': ['programista','developer','frontend','backend','fullstack','javascript','python','java','software','it','devops','kubernetes','docker','cloud'],
  'Finanse': ['księgowy','finanse','rachunkowość','audyt','bankowość','analiza finansowa','inwestycje','fp&a','controlling'],
  'Marketing': ['marketing','social media','kampania','reklama','seo','sem','content'],
  'Sprzedaż': ['sprzedawca','sales','account manager','klient','handel','negocjacje'],
  'HR': ['rekrutacja','hr','kadry','zasoby ludzkie','onboarding'],
  'Logistyka': ['logistyka','transport','magazyn','łańcuch dostaw','spedycja'],
  'Inżynieria': ['inżynier','projektowanie','mechanika','budowa maszyn','automatyka','cnc'],
  'Prawo': ['prawnik','radca prawny','adwokat','prawo','umowa','kodeks'],
  'Zdrowie': ['lekarz','pielęgniarka','medycyna','szpital','pacjent','rehabilitacja'],
  'Edukacja': ['nauczyciel','wykładowca','szkolenie','edukacja','kurs','uczeń'],
  'Consulting': ['konsultant','doradztwo','strategia','analiza biznesowa'],
  'Inne': []
};

app.post('/api/detect-industry', jsonParser, async (req, res) => {
  try {
    const urls = normalizeToArray(req.body?.jobUrls || req.body?.urls);
    if (!urls.length) {
      return res.status(400).json({ error: 'Brak poprawnych linków' });
    }
    const pages = await Promise.all(urls.map(async (url) => {
      try {
        const response = await axios.get(url, { timeout: 8000, headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' } });
        return String(response.data || '').toLowerCase();
      } catch (err) {
        console.warn(`Nie udało się pobrać ${url}:`, err.message);
        return '';
      }
    }));

    let detectedIndustry = '';
    for (const pageText of pages) {
      if (!pageText) continue;
      for (const [industry, keywords] of Object.entries(INDUSTRY_KEYWORDS)) {
        if (keywords.some(keyword => pageText.includes(keyword))) {
          detectedIndustry = industry; break;
        }
      }
      if (detectedIndustry) break;
    }
    return res.json({ industry: detectedIndustry });
  } catch (error) {
    console.error('Błąd /api/detect-industry:', error);
    res.status(500).json({ error: 'Błąd serwera przy wykrywaniu branży' });
  }
});

// ==== ERROR HANDLER ====
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'Plik jest za duży (max 10MB)' });
    }
  }
  console.error('Błąd serwera:', error);
  res.status(500).json({ error: 'Wewnętrzny błąd serwera', details: error.message });
});

// ==== START ====
app.listen(PORT, () => {
  console.log(`🚀 Server działa na porcie ${PORT}`);
});

app.post('/api/contact', jsonParser, contactLimiter, async (req, res) => {
  try {
    const { name, email, message, website } = req.body || {};

    // Honeypot: jeśli bot wypełni "website" → udaj sukces i wyjdź
    if (website && String(website).trim() !== '') {
      return res.json({ ok: true });
    }

    // Walidacje
    if (!name || !email || !message) {
      return res.status(400).json({ error: 'Brak wymaganych pól' });
    }
    if (!validator.isEmail(String(email))) {
      return res.status(400).json({ error: 'Nieprawidłowy adres e-mail' });
    }
    const cleanName = String(name).trim().slice(0, 120);
    const cleanEmail = String(email).trim().slice(0, 180);
    const cleanMsg = String(message).trim().slice(0, 5000);

    // Wyślij e-mail do siebie
    const toAddress = process.env.CONTACT_TO || process.env.SMTP_USER;
    await mailTransport.sendMail({
      from: `"Interview Prep – Formularz" <${process.env.SMTP_USER}>`,
      to: toAddress,
      replyTo: cleanEmail,
      subject: `Nowe zgłoszenie kontaktowe od: ${cleanName}`,
      text: `Imię i nazwisko: ${cleanName}\nE-mail: ${cleanEmail}\n\nWiadomość:\n${cleanMsg}`
    });

    return res.json({ ok: true });
  } catch (err) {
    console.error('❌ /api/contact', err);
    return res.status(500).json({ error: 'Błąd wysyłki wiadomości' });
  }
});
