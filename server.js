/**
 * @fileoverview Serwer Node.js do analizy CV za pomocą Gemini AI.
 * Obsługuje przesyłanie plików, pobieranie ogłoszeń o pracę
 * i generowanie analizy.
 */

const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
require('dotenv').config();

const { GoogleGenerativeAI } = require("@google/generative-ai");
const { JSDOM } = require('jsdom');
const { Readability } = require('@mozilla/readability');

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

// ==== Endpoint: analiza CV z wieloma URL + dodatkowy opis ====
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

