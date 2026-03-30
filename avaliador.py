from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import pandas as pd

# Modelo de embeddings em português/multilíngue
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def semantic_similarity(text1: str, text2: str) -> float:
    emb = model.encode([text1, text2])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(score)

def lexical_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def chatbot_faq_score(faq_answer: str, chatbot_answer: str) -> dict:
    sem_score = semantic_similarity(faq_answer, chatbot_answer)
    lex_score = lexical_similarity(faq_answer, chatbot_answer)

    final_score = 0.7 * sem_score + 0.3 * lex_score

    return {
        "faq_answer": faq_answer,
        "chatbot_answer": chatbot_answer,
        "semantic_score": round(sem_score, 4),
        "lexical_score": round(lex_score, 4),
        "final_score": round(final_score, 4)
    }
# Exemplo
faq = "Você pode solicitar a segunda via do boleto na área do cliente."
chatbot = "A segunda via do boleto pode ser emitida pelo portal do cliente."

resultado = chatbot_faq_score(faq, chatbot)
print(resultado)



# Como interpretar

# Você pode definir faixas assim:

# 0.85 a 1.00 → excelente
# 0.70 a 0.84 → aceitável
# 0.50 a 0.69 → fraco
# abaixo de 0.50 → ruim