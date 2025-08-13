/**
 * @fileoverview Serwer Node.js do analizy CV za pomocą Gemini AI.
 * Obsługuje przesyłanie plików, pobieranie ogłoszeń o pracę
 * i generowanie analizy.
 */

// Usunięte: import helmet from 'helmet';
import * as pdfParse from 'pdf-parse/lib/pdf-parse.js';
import mammoth from 'mammoth';
import fs from 'fs';
import path from 'path';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import axios from 'axios';
import 'dotenv/config';

import { GoogleGenerativeAI } from "@google/generative-ai";
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

import fetch from 'node-fetch';

const app = express();
const PORT = process.env.PORT || 5000;

// ==== Konfiguracja GEMINI ====
if (!process.env.GEMINI_API_KEY) {
    console.error("❌ Brak GEMINI_API_KEY w pliku .env");
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ==== Multer do uploadu plików ====
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, 'uploads/'),
    filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });
if (!fs.existsSync('uploads')) fs.mkdirSync('uploads');

app.use(cors());
app.use(express.json());

// ❌ Usunięto Helmet CSP – backend nie wysyła już default-src 'none'

// ==== Pobieranie oferty pracy ====
const getJobDescriptionWithReadability = async (url) => {
    try {
        const response = await axios.get(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const doc = new JSDOM(response.data, { url });
        const reader = new Readability(doc.window.document);
        const article = reader.parse();

        return article
            ? article.textContent.replace(/\s+/g, ' ').trim()
            : 'Nie udało się pobrać treści ogłoszenia.';
    } catch (error) {
        console.error("❌ Błąd podczas pobierania ogłoszenia:", error.message);
        return 'Nie udało się pobrać treści ogłoszenia.';
    }
};

// ==== Odczyt CV ====
const extractTextFromCV = async (filePath) => {
    const ext = path.extname(filePath).toLowerCase();
    let textContent = '';
    
    if (ext === '.pdf') {
        const dataBuffer = fs.readFileSync(filePath);
        // Zmieniono sposób wywołania funkcji, aby pasował do nowego importu
        const pdfData = await pdfParse(dataBuffer);
        textContent = pdfData.text;
    } else if (ext === '.docx') {
        const result = await mammoth.extractRawText({ path: filePath });
        textContent = result.value;
    } else {
        throw new Error('Obsługiwane są tylko pliki PDF i DOCX.');
    }
    
    return textContent;
};

// ==== Endpoint: analiza CV z wieloma URL + dodatkowy opis + wersje branżowe ====
app.post('/api/analyze-cv-multiple', upload.single('cv'), async (req, res) => {
    try {
        const { jobUrls, plan, additionalDescription, selectedIndustry } = req.body;
        const urls = JSON.parse(jobUrls);

        // Odczyt CV
        const { path: filePath } = req.file;
        let textContent = await extractTextFromCV(filePath);

        // Doklejenie dodatkowego opisu, jeśli jest
        if (additionalDescription && additionalDescription.trim()) {
            textContent += `\n\nDodatkowy opis od kandydata:\n${additionalDescription.trim()}`;
        }

        // Usunięcie pliku po odczycie
        fs.unlink(filePath, (err) => {
            if (err) console.error("⚠️ Błąd przy usuwaniu pliku:", err.message);
        });

        // Pobranie opisów ofert pracy
        const jobDescriptions = await Promise.all(urls.map(getJobDescriptionWithReadability));
        const combinedJobDescriptions = jobDescriptions.join("\n\n---\n\n");

        // Ustalanie liczby pytań zależnie od planu
        const isFree = plan === "free";
        const numSoft = isFree ? 2 : 7;
        const numHard = isFree ? 2 : 10;

        const prompt = `
Jesteś ekspertem HR. Analizujesz CV kandydata, ogłoszenia o pracę oraz (jeśli podano) dodatkowy opis i branżę.

=== Branża ===
${selectedIndustry || 'Nie podano'}

Twoje zadania:
1. Oceń CV ogólnie — mocne i słabe strony i rekomenduj zmiany.
2. Oceń dopasowanie CV do wszystkich ofert (wskaż dopasowania i braki).
3. Wygeneruj ${numSoft} pytań o kompetencje miękkie.
4. Wygeneruj ${numHard} pytań o kompetencje twarde.

Zwróć odpowiedź w JSON:
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
        `;

        // Wywołanie Gemini
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const result = await model.generateContent(prompt);
        const llmResponse = await result.response.text();

        let analysis;
        try {
            const cleanedResponse = llmResponse.replace(/```json|```/g, '').trim();
            analysis = JSON.parse(cleanedResponse);

            if (analysis?.dopasowanie?.mocne_strony && analysis?.dopasowanie?.obszary_do_poprawy) {
                const total = analysis.dopasowanie.mocne_strony.length + analysis.dopasowanie.obszary_do_poprawy.length;
                analysis.dopasowanie_procentowe = total > 0
                    ? Math.round((analysis.dopasowanie.mocne_strony.length / total) * 100)
                    : 0;
            } else {
                analysis.dopasowanie_procentowe = 0;
            }

        } catch (parseError) {
            console.error("Błąd parsowania odpowiedzi z Gemini:", parseError);
            analysis = {
                error: 'Błąd parsowania odpowiedzi z Gemini.',
                rawResponse: llmResponse,
                dopasowanie_procentowe: 0
            };
        }

        res.json({
            status: 'success',
            analysis
        });

    } catch (error) {
        console.error("🔥 Nieoczekiwany błąd:", error);
        res.status(500).json({ error: 'Błąd podczas analizy CV', details: error.message });
    }
});

// --- AUTODETEKCJA BRANŻY --- //
const INDUSTRY_KEYWORDS = {
    'IT': ['programista', 'developer', 'frontend', 'backend', 'fullstack', 'javascript', 'python', 'java', 'software', 'it'],
    'Finanse': ['księgowy', 'finanse', 'rachunkowość', 'audyt', 'bankowość', 'analiza finansowa', 'inwestycje'],
    'Marketing': ['marketing', 'social media', 'kampania', 'reklama', 'SEO', 'SEM', 'content'],
    'Sprzedaż': ['sprzedawca', 'sales', 'account manager', 'klient', 'handel', 'negocjacje'],
    'HR': ['rekrutacja', 'hr', 'kadry', 'zasoby ludzkie', 'onboarding'],
    'Logistyka': ['logistyka', 'transport', 'magazyn', 'łańcuch dostaw', 'spedycja'],
    'Inżynieria': ['inżynier', 'projektowanie', 'mechanika', 'budowa maszyn', 'automatyka', 'CNC'],
    'Prawo': ['prawnik', 'radca prawny', 'adwokat', 'prawo', 'umowa', 'kodeks'],
    'Zdrowie': ['lekarz', 'pielęgniarka', 'medycyna', 'szpital', 'pacjent', 'rehabilitacja'],
    'Edukacja': ['nauczyciel', 'wykładowca', 'szkolenie', 'edukacja', 'kurs', 'uczeń'],
    'Consulting': ['konsultant', 'doradztwo', 'strategia', 'analiza biznesowa'],
    'Inne': []
};

app.post('/api/detect-industry', async (req, res) => {
    const { jobUrls } = req.body;

    if (!jobUrls || !Array.isArray(jobUrls) || jobUrls.length === 0) {
        return res.status(400).json({ error: 'Brak poprawnych linków' });
    }

    try {
        // Pobieramy wszystkie ogłoszenia równolegle
        const fetchPromises = jobUrls.map(async (url) => {
            try {
                const response = await fetch(url, { timeout: 8000 });
                if (!response.ok) {
                    console.warn(`Błąd pobierania ${url}: ${response.status}`);
                    return '';
                }
                const html = await response.text();
                return html.toLowerCase();
            } catch (err) {
                console.warn(`Nie udało się pobrać ${url}:`, err.message);
                return '';
            }
        });

        const pages = await Promise.all(fetchPromises);

        let detectedIndustry = '';

        // Szukamy dopasowania w każdej stronie
        for (const text of pages) {
            if (!text) continue;

            for (const [industry, keywords] of Object.entries(INDUSTRY_KEYWORDS)) {
                if (keywords.some(keyword => text.includes(keyword))) {
                    detectedIndustry = industry;
                    break;
                }
            }
            if (detectedIndustry) break; // jeśli znaleziono, kończymy
        }

        return res.json({ industry: detectedIndustry });
    } catch (error) {
        console.error('Błąd /api/detect-industry:', error);
        res.status(500).json({ error: 'Błąd serwera przy wykrywaniu branży' });
    }
});


app.listen(PORT, () => {
    console.log(`🚀 Server działa na porcie ${PORT}`);
});