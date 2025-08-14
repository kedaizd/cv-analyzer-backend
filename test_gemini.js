import 'dotenv/config';
import { GoogleGenerativeAI } from '@google/generative-ai';

(async () => {
  try {
    if (!process.env.GEMINI_API_KEY) throw new Error('Brak GEMINI_API_KEY w .env');
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const res = await model.generateContent('Napisz jedno słowo: OK');
    console.log('✅ Gemini działa. Odpowiedź:', await res.response.text());
  } catch (e) {
    console.error('❌ Test nieudany:', e.message);
    process.exit(1);
  }
})();
