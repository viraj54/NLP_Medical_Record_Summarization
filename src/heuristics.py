"""Experiment text utilities and heuristic scores; not clinical validators."""
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    if not isinstance(text, str):
        return ''
    return re.sub('\\s+', ' ', text).strip()

def tfidf_extractive_summary(text, n_sentences=5):
    if n_sentences <= 0:
        raise ValueError("n_sentences must be positive")
    """Rank and select sentences with scikit-learn TF-IDF; retain their original order."""
    sentences = re.split('(?<=[.!?])\\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(sentences)
        doc_vector = np.asarray(tfidf_matrix.mean(axis=0))
        scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()
        top_idx = sorted(np.argsort(scores)[-n_sentences:].tolist())
        return ' '.join([sentences[i] for i in top_idx])
    except Exception:
        return ' '.join(sentences[:n_sentences])

MEDICAL_ENTITY_PATTERNS = ['\\b(diagnosis|diagnos\\w+)\\b', '\\b(treatment|treat\\w+)\\b', '\\b(symptom\\w*)\\b', '\\b(medication\\w*|drug\\w*|prescription)\\b', '\\b(patient|hospital|clinic\\w*)\\b', '\\b(blood pressure|heart rate|temperature|oxygen)\\b', '\\b(mg|ml|mcg|mmhg|bpm)\\b', '\\b(surgery|surgical|operation|procedure)\\b', '\\b(cancer|tumor|malignant|benign)\\b', '\\b(test|exam\\w*|scan|mri|ct|xray|ultrasound|biopsy)\\b', '\\b(discharge|admission|emergency)\\b', '\\b(pain|fever|nausea|fatigue|inflammation)\\b']

def extract_medical_entities_regex(text):
    """Fallback: extract medical keywords using regex patterns."""
    text_lower = text.lower()
    entities = set()
    for pattern in MEDICAL_ENTITY_PATTERNS:
        matches = re.findall(pattern, text_lower)
        entities.update(matches)
    return entities

STOPWORDS = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'and', 'or', 'but', 'not', 'no', 'this', 'that', 'it', 'its', 'he', 'she', 'they', 'we', 'i', 'none', 'null', 'true', 'false', 'patient', 'doctor', 'hospital'}

def tokenize_content(text):
    """Lowercase, remove punctuation, filter stopwords — keep content tokens."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def source_token_overlap(prediction, source_note):
    """
    Token-level faithfulness: proportion of content tokens in the
    prediction that also appear in the source note.
    Score = |pred_tokens ∩ source_tokens| / |pred_tokens|
    This lexical score does not detect hallucination, negation errors or wrong relations.
    """
    if not prediction or not source_note:
        return 0.0
    pred_tokens = tokenize_content(prediction)
    source_tokens = set(tokenize_content(source_note))
    if not pred_tokens:
        return 0.0
    overlap = sum((1 for t in pred_tokens if t in source_tokens))
    return overlap / len(pred_tokens)

CLINICAL_CONCEPTS = {'diseases_conditions': ['hypertension', 'diabetes', 'cancer', 'tumor', 'fracture', 'infection', 'pneumonia', 'asthma', 'arthritis', 'anemia', 'sepsis', 'stroke', 'myocardial infarction', 'heart failure', 'renal failure', 'fibrosis', 'necrosis', 'edema', 'stenosis', 'thrombosis', 'embolism', 'hemorrhage', 'osteoporosis', 'osteomalacia', 'dysplasia', 'malignant', 'benign', 'carcinoma', 'lymphoma', 'leukemia', 'disorder', 'syndrome', 'disease'], 'symptoms': ['pain', 'fever', 'nausea', 'vomiting', 'fatigue', 'weakness', 'dyspnea', 'cough', 'headache', 'dizziness', 'swelling', 'inflammation', 'rash', 'bleeding', 'bruising', 'tenderness', 'stiffness', 'numbness', 'paralysis', 'seizure', 'confusion', 'loss of consciousness', 'shortness of breath'], 'medications': ['mg', 'dose', 'tablet', 'capsule', 'injection', 'antibiotic', 'analgesic', 'steroid', 'insulin', 'aspirin', 'metformin', 'lisinopril', 'atorvastatin', 'amoxicillin', 'ibuprofen', 'paracetamol', 'chemotherapy', 'immunotherapy', 'medication', 'drug', 'prescription', 'therapy', 'treatment'], 'procedures': ['surgery', 'biopsy', 'resection', 'excision', 'transplant', 'amputation', 'catheterization', 'dialysis', 'transfusion', 'ventilation', 'intubation', 'anesthesia', 'laparoscopy', 'thoracotomy', 'arthroplasty', 'osteotomy'], 'diagnostics': ['mri', 'ct scan', 'x-ray', 'ultrasound', 'biopsy', 'blood test', 'ecg', 'echocardiography', 'colonoscopy', 'endoscopy', 'spirometry', 'hemoglobin', 'glucose', 'creatinine', 'platelet', 'white blood cell', 'red blood cell', 'culture', 'urinalysis'], 'anatomy': ['heart', 'lung', 'liver', 'kidney', 'brain', 'spine', 'femur', 'tibia', 'humerus', 'pelvis', 'abdomen', 'thorax', 'chest', 'neck', 'hip', 'knee', 'elbow', 'shoulder', 'artery', 'vein', 'nerve', 'muscle', 'bone', 'joint']}

def extract_concepts(text):
    """Extract clinical concepts present in text (case-insensitive substring match)."""
    if not text:
        return set()
    text_lower = text.lower()
    return {concept for concept in ALL_CONCEPTS if concept in text_lower}

def lexicon_concept_f1(prediction, reference):
    """
    Compute custom lexicon concept F1 for a single prediction-reference pair.
    Precision = |pred_concepts ∩ ref_concepts| / |pred_concepts|
    Recall    = |pred_concepts ∩ ref_concepts| / |ref_concepts|
    F1        = 2 * P * R / (P + R)
    """
    pred_concepts = extract_concepts(prediction)
    ref_concepts = extract_concepts(reference)
    if not pred_concepts or not ref_concepts:
        return 0.0
    overlap = len(pred_concepts & ref_concepts)
    precision = overlap / len(pred_concepts)
    recall = overlap / len(ref_concepts)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

ALL_CONCEPTS = {term for terms in CLINICAL_CONCEPTS.values() for term in terms}
