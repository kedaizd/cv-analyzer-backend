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
const require = createRequire(import.meta.url);
// Ładowanie pdf-parse bezpośrednio z lib (omijamy "demo code" z index.js)
const pdfParse = require('pdf-parse/lib/pdf-parse.js');
import mammoth from 'mammoth';

import { GoogleGenerativeAI } from "@google/generative-ai";
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

const app = express();
const PORT = process.env.PORT || 5000;

// ==== Konfiguracja GEMINI ====
if (!process.env.GEMINI_API_KEY) {
  console.error("❌ Brak GEMINI_API_KEY w pliku .env");
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// ==== Limity i przycinanie tekstów ====
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

// ==== Multer do uploadu plików ====
if (!fs.existsSync('uploads')) fs.mkdirSync('uploads', { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.pdf', '.docx'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) cb(null, true);
    else cb(new Error('Dozwolone tylko pliki PDF i DOCX'));
  }
});

// ==== Middleware ====
const allowedOrigins = [
  'https://cv-analyzer-frontend.vercel.app',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  'http://localhost:3000',
  'http://localhost:5000'
];
app.use(cors({ origin: allowedOrigins, credentials: true }));

// Parsowanie JSON tylko w konkretnych endpointach
const jsonParser = express.json({ limit: '1mb' });

// Healthcheck
app.get('/api/health', (req, res) => {
  res.json({ ok: true, hasGeminiKey: !!process.env.GEMINI_API_KEY });
});

// ==== Helpers ====
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

const sanitizeStringArray = (arr, maxLen) => {
  if (!Array.isArray(arr)) return [];
  return arr.map(x => clampText(String(x || '').trim(), maxLen)).filter(Boolean);
};

// Pobiera opis ogłoszenia
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
    console.error("❌ Błąd podczas pobierania ogłoszenia:", error.message);
    return 'Nie udało się pobrać treści ogłoszenia.';
  }
};

// Odczyt CV (PDF: pdf-parse, DOCX: mammoth)
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

// Pobranie treści z Gemini z fallbackiem na text/plain
const getLLMOutputString = async (resp) => {
  try {
    const parts = resp?.candidates?.[0]?.content?.parts || [];
    const jsonInline = parts.find(
      p => p?.inlineData && /json/i.test(p.inlineData.mimeType || '')
    );
    if (jsonInline?.inlineData?.data) {
      return Buffer.from(jsonInline.inlineData.data, 'base64').toString('utf-8');
    }
  } catch (e) {
    // ignore
  }
  try {
    return await resp.text();
  } catch {
    return '';
  }
};

// Bezpieczne wywołanie Gemini z retry w trybie text/plain
const generateWithFallback = async (prompt) => {
  // Pierwsza próba: normalnie
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: LIMITS.maxOutputTokens
      }
    });
    const result = await model.generateContent(prompt);
    return await getLLMOutputString(result.response);
  } catch (e) {
    const msg = String(e?.message || e);
    if (msg.includes('valid JSON') || msg.toLowerCase().includes('json')) {
      console.warn('Gemini JSON error - retry w text/plain:', msg);
      const modelPlain = genAI.getGenerativeModel({
        model: "gemini-1.5-flash",
        generationConfig: {
          responseMimeType: "text/plain",
          temperature: 0.2,
          maxOutputTokens: LIMITS.maxOutputTokens
        }
      });
      const retry = await modelPlain.generateContent(prompt);
      // Tutaj wymuszamy tekst
      try {
        return await retry.response.text();
      } catch {
        const parts = retry?.response?.candidates?.[0]?.content?.parts || [];
        return parts.map(p => p?.text || '').join('') || '';
      }
    }
    // Inny błąd – przekaż dalej
    throw e;
  }
};

// ==== Endpoint: analiza CV ====
app.post('/api/analyze-cv-multiple', upload.single('cv'), async (req, res) => {
    // --- Dodaj te logi na początku funkcji ---
  console.log("-------------------------------------");
  console.log("Odebrano żądanie do /api/analyze-cv-multiple");
  console.log("Plik CV:", req.file ? req.file.originalname : "Brak pliku");
  console.log("Body żądania:", req.body);
  console.log("-------------------------------------");

  let uploadedFilePath;
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Brak przesłanego pliku CV' });
    }

    console.log("🚀 Rozpoczynam analizę CV...");
    console.log("Klucz API Gemini odczytany:", !!process.env.GEMINI_API_KEY);

    if (!process.env.GEMINI_API_KEY) {
      console.error("🔥 Błąd: Klucz API Gemini nie jest ustawiony!");
      return res.status(400).json({
        error: 'Brak GEMINI_API_KEY w .env — ustaw go, aby uruchomić analizę.',
        hint: 'Dodaj GEMINI_API_KEY=... do pliku .env i zrestartuj serwer.'
      });
    }

    uploadedFilePath = req.file.path;

    const urls = normalizeToArray(req.body?.urls);
    const additionalDescription = typeof req.body?.additionalDescription === 'string'
      ? clampText(req.body.additionalDescription, LIMITS.additionalDescriptionChars)
      : '';
    const plan = typeof req.body?.plan === 'string' ? req.body.plan : 'free';
    const selectedIndustry = typeof req.body?.selectedIndustry === 'string'
      ? req.body.selectedIndustry
      : '';

    let textContent = await extractTextFromCV(uploadedFilePath);

    if (additionalDescription && additionalDescription.trim()) {
      textContent += `\n\nDodatkowy opis od kandydata:\n${additionalDescription.trim()}`;
      textContent = clampText(textContent, LIMITS.cvChars + LIMITS.additionalDescriptionChars);
    }

    const jobDescriptions = urls.length
      ? await Promise.all(urls.map(getJobDescriptionWithReadability))
      : [];
    let combinedJobDescriptions = jobDescriptions.length
      ? jobDescriptions.join("\n\n---\n\n")
      : 'Brak ogłoszeń lub nie udało się pobrać treści.';
    combinedJobDescriptions = clampText(combinedJobDescriptions, LIMITS.jobTotalChars);

    const isFree = String(plan).toLowerCase() === "free";
    const numSoft = isFree ? 2 : 7;
    const numHard = isFree ? 2 : 10;

    const prompt = `
Jesteś ekspertem HR. Analizujesz CV kandydata, ogłoszenia o pracę oraz (jeśli podano) dodatkowy opis i branżę.

=== Branża ===
${selectedIndustry || 'Nie podano'}

Zadania:
- Oceń CV ogólnie — mocne i słabe strony i rekomenduj zmiany.
- Oceń dopasowanie CV do wszystkich ofert (wskaż dopasowania i braki).
- Wygeneruj ${numSoft} pytań o kompetencje miękkie.
- Wygeneruj ${numHard} pytań o kompetencje twarde.

Ograniczenia długości odpowiedzi:
- "podsumowanie" maks. ${LIMITS.summaryChars} znaków.
- Każda pozycja w listach maks. ${LIMITS.itemChars} znaków.
- Dokładnie ${numSoft} pytań miękkich i ${numHard} pytań twardych.

Zwróć odpowiedź WYŁĄCZNIE jako czysty JSON (bez markdown, bez komentarzy), w strukturze:
{
  "podsumowanie": "...",
  "dopasowanie": {
    "mocne_strony": ["..."],
    "obszary_do_poprawy": ["..."]
  },
  "pytania": {
    "kompetencje_miekkie": ["..."],
    "kompetencje_twarde": ["..."]
  }
}

=== CV (z opisem kandydata) ===
${textContent}

=== OGŁOSZENIA ===
${combinedJobDescriptions}
`.trim();

    // Wywołanie Gemini z fallbackiem
    const llmRaw = await generateWithFallback(prompt);

    let analysis;
    try {
      const parsed = extractAndParseJSON(llmRaw);
      analysis = ensureAnalysisShape(parsed);

      analysis.podsumowanie = clampText(analysis.podsumowanie, LIMITS.summaryChars);
      analysis.dopasowanie.mocne_strony = sanitizeStringArray(analysis.dopasowanie.mocne_strony, LIMITS.itemChars);
      analysis.dopasowanie.obszary_do_poprawy = sanitizeStringArray(analysis.dopasowanie.obszary_do_poprawy, LIMITS.itemChars);
      analysis.pytania.kompetencje_miekkie = sanitizeStringArray(analysis.pytania.kompetencje_miekkie, LIMITS.itemChars).slice(0, numSoft);
      analysis.pytania.kompetencje_twarde = sanitizeStringArray(analysis.pytania.kompetencje_twarde, LIMITS.itemChars).slice(0, numHard);
    } catch (parseError) {
      console.error("Błąd parsowania odpowiedzi z Gemini:", parseError);
      analysis = ensureAnalysisShape({});
      analysis.podsumowanie = 'Nie udało się poprawnie sparsować odpowiedzi modelu. Poniżej surowa odpowiedź (skrót).';
      analysis.rawResponse = String(llmRaw || '').slice(0, 2000);
    }

    const total = analysis.dopasowanie.mocne_strony.length + analysis.dopasowanie.obszary_do_poprawy.length;
    analysis.dopasowanie_procentowe = total > 0
      ? Math.round((analysis.dopasowanie.mocne_strony.length / total) * 100)
      : 0;

    return res.json({ status: 'success', analysis });

  } catch (error) {
    console.error("🔥 Nieoczekiwany błąd:", error);
    return res.status(500).json({ error: 'Błąd podczas analizy CV', details: error.message });
  } finally {
    if (uploadedFilePath) {
      fs.unlink(uploadedFilePath, (err) => {
        if (err) console.error("Błąd usuwania pliku:", err);
      });
    }
  }
});

// --- AUTODETEKCJA BRANŻY --- //
const INDUSTRY_KEYWORDS = {
  'IT': ['programista', 'developer', 'frontend', 'backend', 'fullstack', 'javascript', 'python', 'java', 'software', 'it'],
  'Finanse': ['księgowy', 'finanse', 'rachunkowość', 'audyt', 'bankowość', 'analiza finansowa', 'inwestycje'],
  'Marketing': ['marketing', 'social media', 'kampania', 'reklama', 'seo', 'sem', 'content'],
  'Sprzedaż': ['sprzedawca', 'sales', 'account manager', 'klient', 'handel', 'negocjacje'],
  'HR': ['rekrutacja', 'hr', 'kadry', 'zasoby ludzkie', 'onboarding'],
  'Logistyka': ['logistyka', 'transport', 'magazyn', 'łańcuch dostaw', 'spedycja'],
  'Inżynieria': ['inżynier', 'projektowanie', 'mechanika', 'budowa maszyn', 'automatyka', 'cnc'],
  'Prawo': ['prawnik', 'radca prawny', 'adwokat', 'prawo', 'umowa', 'kodeks'],
  'Zdrowie': ['lekarz', 'pielęgniarka', 'medycyna', 'szpital', 'pacjent', 'rehabilitacja'],
  'Edukacja': ['nauczyciel', 'wykładowca', 'szkolenie', 'edukacja', 'kurs', 'uczeń'],
  'Consulting': ['konsultant', 'doradztwo', 'strategia', 'analiza biznesowa'],
  'Inne': []
};

app.post('/api/detect-industry', jsonParser, async (req, res) => {
  try {
    const urls = normalizeToArray(req.body?.jobUrls || req.body?.urls);
    if (!urls.length) {
      return res.status(400).json({ error: 'Brak poprawnych linków' });
    }

    const fetchPromises = urls.map(async (url) => {
      try {
        const response = await axios.get(url, {
          timeout: 8000,
          headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const html = String(response.data || '').toLowerCase();
        return html;
      } catch (err) {
        console.warn(`Nie udało się pobrać ${url}:`, err.message);
        return '';
      }
    });

    const pages = await Promise.all(fetchPromises);
    let detectedIndustry = '';

    for (const pageText of pages) {
      if (!pageText) continue;
      for (const [industry, keywords] of Object.entries(INDUSTRY_KEYWORDS)) {
        if (keywords.some(keyword => pageText.includes(keyword))) {
          detectedIndustry = industry;
          break;
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

// Start serwera
app.listen(PORT, () => {
  console.log(`🚀 Server działa na porcie ${PORT}`);
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'Plik jest za duży (max 10MB)' });
    }
  }
  console.error('Błąd serwera:', error);
  res.status(500).json({ error: 'Wewnętrzny błąd serwera', details: error.message });
});