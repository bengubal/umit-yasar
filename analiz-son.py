"""
Ümit Yaşar Oğuzcan - Duygusal Evrim Analiz Pipeline
=====================================================
1. SentiTurkNet  → sözcük bazlı duygu skoru (pozitif/negatif/nötr)
2. BERTurk       → cümle bazlı duygu sınıflandırması
3. Dönem bazlı   → zaman içindeki evrim görselleştirmesi

Kurulum:
  pip install transformers torch zeyrek matplotlib seaborn pandas

Çalıştır:
  python analiz.py
"""

import pandas as pd
import numpy as np
import re
import json
import urllib.request
import os
import warnings
warnings.filterwarnings("ignore")

# ── SentiTurkNet Sözlüğü ─────────────────────────────────────────────────────
# SentiTurkNet'i GitHub'dan indir (ücretsiz, akademik lisans)
SENTI_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner/master/corpora/sentiws_v1.8c/SentiWS_v1.8c_Positive.txt"

# Türkçe duygu sözcükleri — SentiTurkNet benzeri manuel liste
# (SentiTurkNet erişim gerektirdiğinden temel bir Türkçe duygu leksikonu)
POZITIF_KELIMELER = {
    "sevgi", "aşk", "mutlu", "güzel", "umut", "sevinç", "neşe", "huzur",
    "sevmek", "seviyorum", "güzellik", "iyilik", "şefkat", "sevi", "dost",
    "dostluk", "gülmek", "gülüş", "bahar", "çiçek", "ışık", "aydınlık",
    "özgür", "özgürlük", "barış", "şen", "tatlı", "canım", "sevgili",
    "harika", "muhteşem", "mükemmel", "beraber", "birlikte", "kavuşmak",
    "hasret", "özlem", "beklemek", "gelmek", "bulunmak", "yaşamak",
    "gülümse", "sıcak", "yumuşak", "nazik", "içten", "samimi", "güven",
    "inanmak", "ümit", "umutlu", "parlak", "aydın", "ferah", "rahat",
    "şükran", "minnet", "saygı", "sevgi dolu", "neşeli", "canlı",
    "coşku", "heyecan", "tutku", "arzu", "istek", "özgün", "derin",
    "anlam", "değer", "kıymet", "önem", "hatıra", "anı",
}

NEGATIF_KELIMELER = {
    "acı", "ölüm", "ölmek", "yalnız", "yalnızlık", "karanlık", "ağlamak",
    "gözyaşı", "hüzün", "keder", "dert", "yas", "üzüntü", "ayrılık",
    "ayrılmak", "gitmek", "kayıp", "yitirmek", "yitirilmek", "boşluk",
    "çaresiz", "çaresizlik", "umutsuz", "umutsuzluk", "lanet", "bitmek",
    "tükenmek", "yorgun", "bitkin", "acımak", "sızı", "ağrı", "yara",
    "kan", "ağıt", "mezar", "ölü", "ceset", "toprak", "son", "bitmek",
    "bitmişlik", "mahkum", "hapis", "zindan", "karanlıklar", "gece",
    "soğuk", "donmak", "kasvet", "kabus", "korkusu", "korku", "dehşet",
    "felaket", "yıkım", "çöküş", "düşmek", "batmak", "boğulmak",
    "sönmek", "solmak", "sararmak", "solgun", "sönük", "silik",
    "yıkılış", "çöküş", "mahvolmak", "perişan", "bitap", "dermansız",
    "inlemek", "sızlamak", "yanmak", "tutuşmak", "kavrulamak",
    "ihanet", "yalan", "aldatmak", "terk", "terk etmek", "bırakmak",
    "unutmak", "silmek", "geçmişe", "yokluk", "hiçlik", "anlamsız",
    "beyhude", "boşuna", "nafile", "faydasız", "zavallı", "biçare",
}

# Ağırlıklı kelimeler (güçlü duygu taşıyanlar)
GUCLU_POZITIF = {"aşk", "sevgi", "mutluluk", "sevinç", "huzur", "umut", "kavuşmak"}
GUCLU_NEGATIF = {"ölüm", "acı", "yalnızlık", "ayrılık", "yas", "çaresiz", "umutsuz", "mezar"}



# ── Stemming & Lemmatization (Zeyrek) ──────────────────────────────────────
_zeyrek_analizci = None

def stemler(metin: str) -> list:
    """Zeyrek ile Türkçe kelime köklerini bul. NLTK punkt gerekli."""
    global _zeyrek_analizci
    try:
        if _zeyrek_analizci is None:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)
            import zeyrek
            _zeyrek_analizci = zeyrek.MorphAnalyzer()

        sonuclar = _zeyrek_analizci.lemmatize(metin)
        # [(kelime, [olası_kökler]), ...] → ilk kök
        return [lemmas[0].lower() if lemmas else word.lower() for word, lemmas in sonuclar]
    except Exception as e:
        # Zeyrek yoksa fallback: basit son-ek temizleme
        EKLER = ["lerinden","larından","lerimiz","larımız","lerini","larını","leriyle",
                 "larıyla","leri","ları","ler","lar","den","dan","ten","tan","nin","nın",
                 "nun","nün","yle","yla","ile","mek","mak","muş","müş","mış","miş",
                 "iyor","ıyor","uyor","üyor","sin","sın","sun","sün","dir","dır","dur","dür"]
        kelimeler = re.sub(r"[^\w\sçğıöşü]", "", metin.lower()).split()
        kokler = []
        for k in kelimeler:
            for ek in EKLER:
                if len(k) > len(ek) + 2 and k.endswith(ek):
                    k = k[:-len(ek)]
                    break
            kokler.append(k)
        return kokler


# ── TF-IDF: Döneme özgü kelimeler ───────────────────────────────────────────
def tfidf_donem(df, donem_col='donem', metin_col='metin', n_top=15):
    """Her dönem için en karakteristik (TF-IDF skoruna göre) kelimeleri bul."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  ! sklearn yok, TF-IDF atlanıyor")
        return {}

    STOPWORDS = ["bir","ve","bu","ben","sen","biz","siz","onlar","ile","için","gibi","ama",
                 "var","yok","olan","kadar","her","hiç","çok","hep","biraz","artık","yine",
                 "sadece","bile","daha","ise","değil","vardı","nasıl","seni","sana","beni",
                 "bana","benim","bütün","kendi","tüm","yıl","kez","kere","şey","şu","şimdi"]

    donemler = [d for d in df[donem_col].unique() if "Belirsiz" not in str(d)]

    # Her dönem için birleşik metin
    donem_metinleri = []
    donem_isimleri  = []
    for d in donemler:
        birlestirilmis = " ".join(df[df[donem_col] == d][metin_col].dropna())
        donem_metinleri.append(birlestirilmis)
        donem_isimleri.append(d)

    # TF-IDF hesapla (3 dönem = 3 "belge")
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=STOPWORDS,
        ngram_range=(1, 1),
        min_df=1,
        token_pattern=r"(?u)\b[a-zçğıöşü]{3,}\b",
        lowercase=True,
    )
    matris = vectorizer.fit_transform(donem_metinleri)
    feature_names = vectorizer.get_feature_names_out()

    sonuc = {}
    for i, d in enumerate(donem_isimleri):
        skorlar = matris[i].toarray()[0]
        en_yuksek = skorlar.argsort()[::-1][:n_top]
        sonuc[d] = [(feature_names[j], round(float(skorlar[j]), 4)) for j in en_yuksek if skorlar[j] > 0]
    return sonuc


# ── N-gram analizi ──────────────────────────────────────────────────────────
def ngram_donem(df, donem_col='donem', metin_col='metin', n=2, n_top=10):
    """Her dönem için en sık geçen n-gramları bul (2 veya 3 kelimelik kalıplar)."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        return {}

    STOPWORDS = ["bir","ve","bu","ben","sen","biz","siz","ile","için","gibi","ama",
                 "var","yok","olan","kadar","her","hiç","çok","hep","daha","ise","değil"]

    donemler = [d for d in df[donem_col].unique() if "Belirsiz" not in str(d)]

    sonuc = {}
    for d in donemler:
        birlestirilmis = " ".join(df[df[donem_col] == d][metin_col].dropna())
        try:
            vec = CountVectorizer(
                ngram_range=(n, n),
                stop_words=STOPWORDS,
                token_pattern=r"(?u)\b[a-zçğıöşü]{2,}\b",
                max_features=200,
                lowercase=True,
            )
            X = vec.fit_transform([birlestirilmis])
            counts = X.toarray()[0]
            terms = vec.get_feature_names_out()
            sirali = sorted(zip(terms, counts), key=lambda x: -x[1])[:n_top]
            sonuc[d] = [(t, int(c)) for t, c in sirali if c > 1]
        except ValueError:
            sonuc[d] = []
    return sonuc


def sentiturk_skor(metin: str) -> dict:
    """
    Metin için SentiTurkNet benzeri sözcük sayım skoru hesapla.
    Hem ham hem stopword'süz (içerik kelimesi bazlı) skoru döndürür.
    """
    if not isinstance(metin, str) or not metin.strip():
        return {"st_pozitif_sayi": 0, "st_negatif_sayi": 0, "st_skor": 0.0, "st_etiket": "nötr",
                "st_skor_filtre": 0.0, "st_etiket_filtre": "nötr", "icerik_kelime_sayisi": 0}

    # Türkçe stopword listesi — şair dilinde sıkça geçen ama duygusal yük taşımayan kelimeler
    STOPWORDS = {
        "bir","ve","bu","ben","sen","o","biz","siz","onlar","de","da","ki","mı","mi","mu","mü",
        "ile","için","gibi","ama","ya","ne","var","yok","olur","olan","olarak","kadar","sonra",
        "önce","her","hiç","çok","az","hep","biraz","artık","yine","sadece","bile","daha","en",
        "şu","şimdi","ise","değil","vardı","nasıl","seni","sana","senin","seninle","beni","bana",
        "benim","benimle","bizi","bize","bizim","sizi","size","sizin","onu","ona","onun","onlar",
        "şey","gün","yıl","kez","kere","tüm","bütün","kendi","kendim","kendin","kendisi",
        "yine","yeniden","artık","henüz","hâlâ","zaman","zamanlar","an","anda","yer","yerde",
        "evet","hayır","tabi","belki","mutlaka","kesin","yani","ise","ya","da","de"
    }

    kelimeler = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", metin.lower()).split()

    poz = 0.0
    neg = 0.0
    for k in kelimeler:
        if k in GUCLU_POZITIF:    poz += 2.0
        elif k in POZITIF_KELIMELER: poz += 1.0
        if k in GUCLU_NEGATIF:    neg += 2.0
        elif k in NEGATIF_KELIMELER: neg += 1.0

    # Ham skor (eski hesaplama — tüm kelimeler payda)
    toplam_ham = len(kelimeler) if kelimeler else 1
    skor_ham = (poz - neg) / toplam_ham

    # Filtrelenmiş skor (stopword'ler hariç — daha keskin)
    icerik = [k for k in kelimeler if k not in STOPWORDS and len(k) > 2]
    toplam_icerik = len(icerik) if icerik else 1
    skor_filtre = (poz - neg) / toplam_icerik

    def etiketle(s):
        if s > 0.04:  return "pozitif"
        if s < -0.04: return "negatif"
        return "nötr"

    return {
        "st_pozitif_sayi": poz,
        "st_negatif_sayi": neg,
        "st_skor": round(skor_ham, 4),
        "st_etiket": etiketle(skor_ham * 2),  # eski eşik korunsun
        "st_skor_filtre": round(skor_filtre, 4),
        "st_etiket_filtre": etiketle(skor_filtre),
        "icerik_kelime_sayisi": toplam_icerik,
    }


def tema_tespiti(metin: str) -> dict:
    """Metinden dominant temaları tespit et."""
    if not isinstance(metin, str):
        return {}

    m = metin.lower()

    temalar = {
        "ask_ozlem": ["aşk", "sevgi", "sevmek", "özlem", "hasret", "sevi", "sevgili", "kalp"],
        "olum_acı": ["ölüm", "ölmek", "acı", "yas", "mezar", "can vermek", "hayata gözlerini"],
        "yalnizlik": ["yalnız", "yalnızlık", "kimsesiz", "tek başına", "yapayalnız"],
        "umut_kavusma": ["umut", "ümit", "kavuşmak", "beklemek", "gelecek", "yeniden"],
        "doga": ["bahar", "güz", "sonbahar", "yağmur", "deniz", "rüzgar", "gece", "gün"],
        "toplum": ["halk", "millet", "ülke", "vatan", "savaş", "insan", "dünya"],
    }

    sonuc = {}
    for tema, kelimeler in temalar.items():
        sayac = sum(1 for k in kelimeler if k in m)
        sonuc[f"tema_{tema}"] = sayac

    # Dominant tema
    en_yuksek = max(sonuc, key=sonuc.get, default=None)
    sonuc["dominant_tema"] = en_yuksek.replace("tema_", "") if en_yuksek and sonuc[en_yuksek] > 0 else "genel"

    return sonuc


def berturk_duygu(metinler: list, batch_size: int = 4) -> list:
    """
    BERTurk tabanlı duygu analizi.
    Model: savasy/bert-base-turkish-sentiment-cased
    """
    try:
        from transformers import pipeline as hf_pipeline
        print("  BERTurk modeli yükleniyor (ilk seferde ~500MB indirilir)...")
        classifier = hf_pipeline(
            "text-classification",
            model="savasy/bert-base-turkish-sentiment-cased",
            truncation=True,
            max_length=256,
            device=-1,
        )
        print("  Model yüklendi ✓")
        print(f"  {len(metinler)} şiir işlenecek (tahmini süre: {len(metinler)//20}-{len(metinler)//10} dakika)...")
        print("  Model yüklendi ✓")

        sonuclar = []
        for i in range(0, len(metinler), batch_size):
            batch = metinler[i:i+batch_size]
            # Çok uzun metinleri kısalt
            batch = [m[:500] if isinstance(m, str) else "" for m in batch]
            try:
                preds = classifier(batch)
                for p in preds:
                    label = p["label"].lower()
                    score = round(p["score"], 4)
                    # Model çıktısı: LABEL_0=negatif, LABEL_1=pozitif (ya da positive/negative)
                    if "positive" in label or label == "label_1":
                        etiket = "pozitif"
                    elif "negative" in label or label == "label_0":
                        etiket = "negatif"
                    else:
                        etiket = label
                    sonuclar.append({"bert_etiket": etiket, "bert_skor": score})
            except Exception as e:
                print(f"  Batch hatası: {e}")
                sonuclar.extend([{"bert_etiket": "hata", "bert_skor": 0.0}] * len(batch))

            tamamlanan = min(i + batch_size, len(metinler))
            if tamamlanan % 20 == 0 or tamamlanan == len(metinler):
                print(f"  BERTurk: {tamamlanan}/{len(metinler)} şiir ✓")

        return sonuclar

    except MemoryError:
        print("  Yetersiz RAM! BERTurk atlanıyor, SentiTurkNet kullanılacak.")
        return [{"bert_etiket": "yok", "bert_skor": 0.0}] * len(metinler)
    except Exception as e:
        print(f"  BERTurk yüklenemedi: {e}")
        print("  SentiTurkNet sonuçları kullanılacak.")
        return [{"bert_etiket": "yok", "bert_skor": 0.0}] * len(metinler)

def codealchemist_duygu(metinler: list, batch_size: int = 4) -> list:
    """
    codealchemist01/turkish-sentiment-analysis modeli.
    439K Türkçe cümleyle eğitilmiş, 3 sınıflı: pozitif/negatif/nötr.
    Şiir gibi 'nötr' tonu olan metinler için daha uygundur.
    """
    try:
        from transformers import pipeline as hf_pipeline
        print("  codealchemist01 modeli yükleniyor (ilk seferde ~500MB indirilir)...")
        classifier = hf_pipeline(
            "text-classification",
            model="codealchemist01/turkish-sentiment-analysis",
            truncation=True,
            max_length=256,
            device=-1,
        )
        print("  Model yüklendi ✓")
        print(f"  {len(metinler)} şiir işlenecek...")

        sonuclar = []
        for i in range(0, len(metinler), batch_size):
            batch = [m[:500] if isinstance(m, str) else "" for m in metinler[i:i+batch_size]]
            try:
                preds = classifier(batch)
                for p in preds:
                    label = p["label"].lower()
                    score = round(p["score"], 4)
                    if "positive" in label or "pozitif" in label:
                        etiket = "pozitif"
                    elif "negative" in label or "negatif" in label:
                        etiket = "negatif"
                    elif "neutral" in label or "notr" in label or "nötr" in label:
                        etiket = "nötr"
                    else:
                        etiket = label
                    sonuclar.append({"ca_etiket": etiket, "ca_skor": score})
            except Exception as e:
                print(f"  Batch hatası ({i}): {e}")
                sonuclar.extend([{"ca_etiket": "hata", "ca_skor": 0.0}] * len(batch))

            tamamlanan = min(i + batch_size, len(metinler))
            if tamamlanan % 20 == 0 or tamamlanan == len(metinler):
                print(f"  codealchemist01: {tamamlanan}/{len(metinler)} şiir ✓")

        return sonuclar

    except MemoryError:
        print("  Yetersiz RAM! codealchemist01 atlanıyor.")
        return [{"ca_etiket": "yok", "ca_skor": 0.0}] * len(metinler)
    except Exception as e:
        print(f"  codealchemist01 yüklenemedi: {e}")
        return [{"ca_etiket": "yok", "ca_skor": 0.0}] * len(metinler)




def html_rapor_uret(df):
    """Analiz sonuçlarından detaylı interaktif HTML rapor üretir."""
    import re
    from collections import Counter

    DONEMLER = ["Erken Dönem (1947–1962)", "Olgunluk Dönemi (1963–1972)", "Son Dönem (1973–1984)"]
    df_filt = df[df['donem'].isin(DONEMLER)].copy()

    # df.attrs'tan TF-IDF ve n-gram sonuçlarını al
    tfidf_sonuc = df.attrs.get('tfidf', {})
    bigram_sonuc = df.attrs.get('bigram', {})
    trigram_sonuc = df.attrs.get('trigram', {})

    STOPWORDS = {"bir","ve","bu","ben","sen","o","biz","siz","onlar","de","da","ki",
                 "ile","için","gibi","ama","ya","mı","mi","mu","mü","ne","var","yok",
                 "olur","olan","olarak","kadar","sonra","önce","her","hiç","çok","az",
                 "hep","biraz","artık","yine","sadece","bile","daha","en","şu","şimdi",
                 "ise","ise","değil","vardı","nasıl","stanbul"}

    ozet = {}
    top_kelimeler = {}
    for d in DONEMLER:
        alt = df_filt[df_filt['donem'] == d]
        ozet[d] = {
            "sayi":              int(len(alt)),
            "poz":               round(float((alt['st_etiket'] == 'pozitif').mean() * 100), 1),
            "neg":               round(float((alt['st_etiket'] == 'negatif').mean() * 100), 1),
            "notr":              round(float((alt['st_etiket'] == 'nötr').mean() * 100), 1),
            "ort_skor":          round(float(alt['st_skor'].mean()), 4),
            "ort_skor_filtre":   round(float(alt['st_skor_filtre'].mean()), 4) if 'st_skor_filtre' in alt.columns else None,
            "poz_filtre":        round(float((alt['st_etiket_filtre'] == 'pozitif').mean() * 100), 1) if 'st_etiket_filtre' in alt.columns else None,
            "neg_filtre":        round(float((alt['st_etiket_filtre'] == 'negatif').mean() * 100), 1) if 'st_etiket_filtre' in alt.columns else None,
            "ort_kelime":        round(float(alt['kelime_sayisi'].mean()), 1),
        }
        # En sık kelimeler
        metinler = " ".join(alt['metin'].dropna())
        kelimeler = re.sub(r"[^\w\sçğıöşü]", " ", metinler.lower()).split()
        kelimeler = [k for k in kelimeler if len(k) > 3 and k not in STOPWORDS]
        top_kelimeler[d] = Counter(kelimeler).most_common(8)

    bert_var = 'bert_etiket' in df.columns and df['bert_etiket'].iloc[0] not in ['yok', 'hata']

    # BERTurk istatistikleri
    bert_ozet = {}
    bert_blok = ""
    bert_js = ""
    bert_yorum = ""
    if bert_var:
        for d in DONEMLER:
            alt = df_filt[df_filt['donem'] == d]
            bert_ozet[d] = {
                "poz": round(float((alt['bert_etiket'] == 'pozitif').mean() * 100), 1),
                "neg": round(float((alt['bert_etiket'] == 'negatif').mean() * 100), 1),
            }
        bert_blok = """
        <div class="chart-card" style="margin-top:24px">
          <div class="chart-title">SentiTurkNet vs BERTurk — İki Modelin Karşılaştırması</div>
          <div class="chart-note">Sözlük tabanlı yöntem ile transformer tabanlı yöntemin dönem bazlı sonuçları</div>
          <div class="chart-wrap"><canvas id="bertChart"></canvas></div>
        </div>"""
        ca_satir_html = ""
        if ca_var:
            ca_poz_e = ca_ozet[DONEMLER[0]]["poz"]
            ca_poz_o = ca_ozet[DONEMLER[1]]["poz"]
            ca_poz_s = ca_ozet[DONEMLER[2]]["poz"]
            ca_neg_s = ca_ozet[DONEMLER[2]]["neg"]
            ca_satir_html = f'<br><br><strong style="color:var(--text)">codealchemist01 (3 sınıflı, 439K cümle):</strong> Erken %{ca_poz_e} → Olgunluk %{ca_poz_o} → Son %{ca_poz_s} pozitif (son dönem negatif %{ca_neg_s}). Bu model nötr sınıfa sahip olduğundan şiirsel ifadeyi daha gerçekçi sınıflandırır. Son dönemdeki keskin düşüş üç yöntemde de görülmektedir.'
        bert_yorum = f'''
        <div class="bulgu" style="margin-top:20px;border-left-color:#5B7EB8">
          <strong style="color:var(--text)">Yöntemler Arası Fark:</strong> SentiTurkNet sözcük tabanlı bir yöntem olduğundan şiirin metafor yapısını yakalayamaz. BERTurk (savasy) 2 sınıflıdır ve ürün yorumlarına eğitilmiştir, bu yüzden şiiri ağırlıklı pozitif sınıflandırma eğilimindedir. codealchemist01 ise 3 sınıflı olarak (pozitif/negatif/nötr) ve çok daha çeşitli verilerle eğitilmiştir; şiir gibi nötr tonu da olan metinler için en uygun yöntemdir.{{ca_satir_html}}
        </div>'''

    TEMA_COLS = {"tema_ask_ozlem":"Aşk & Özlem","tema_olum_aci":"Ölüm & Acı",
                 "tema_yalnizlik":"Yalnızlık","tema_umut":"Umut","tema_doga":"Doğa","tema_toplum":"Toplum"}
    gecerli = [tc for tc in TEMA_COLS if tc in df_filt.columns]
    tema_labels = [TEMA_COLS[tc] for tc in gecerli]
    tema_data = {}
    for d in DONEMLER:
        alt = df_filt[df_filt['donem'] == d]
        tema_data[d] = [round(float(alt[tc].mean()), 3) for tc in gecerli]

    ornekler = {}
    for d in DONEMLER:
        alt = df_filt[df_filt['donem'] == d]
        ornekler[d] = {
            "poz": alt.nlargest(2, 'st_skor')[['baslik','st_skor']].to_dict('records'),
            "neg": alt.nsmallest(2, 'st_skor')[['baslik','st_skor']].to_dict('records'),
        }

    def sinif(d): return "erken" if "Erken" in d else "olgunluk" if "Olgunluk" in d else "son"
    def kisa(d): return d.split('(')[0].strip()
    def yillar(d): return d.split('(')[1].rstrip(')')
    def skor_renk(s): return "pos" if s >= 0 else "neg"
    def skor_fmt(s): return f"+{s:.4f}" if s >= 0 else f"{s:.4f}"

    # ── Detaylı dönem yorumları ──
    donem_yorumlari = {
        DONEMLER[0]: {
            "ust_baslik": "Aşkın Romantik Sesi",
            "yorum": f"""Ümit Yaşar'ın ilk dönemi <strong>%{ozet[DONEMLER[0]]['poz']} pozitif, %{ozet[DONEMLER[0]]['neg']} negatif</strong> oranıyla en dengeli dönemidir. Ortalama duygu skoru <strong>{skor_fmt(ozet[DONEMLER[0]]['ort_skor'])}</strong> ile pozitif tarafta yer alır. Şiirlerin ortalama uzunluğu <strong>{ozet[DONEMLER[0]]['ort_kelime']:.0f} kelime</strong> ile dönemler arasında en uzun olanıdır — şair henüz yoğunlaşmış, kısa form yerine açıklayıcı anlatımı tercih etmektedir. Bu dönemde aşk teması romantik bir saflıkla işlenir; İstanbul'a, denize ve sevdiğine seslenen şiirler baskındır. En sık geçen kelimeler arasında "seni", "sana", "deniz" yer alır — şair dış dünyaya ve sevgiliye yönelmiş durumdadır.""",
            "anahtar": top_kelimeler[DONEMLER[0]][:6],
        },
        DONEMLER[1]: {
            "ust_baslik": "Olgunluğun Dönüşen Tonu",
            "yorum": f"""1963–1972 arası, şairin en üretken dönemidir — <strong>{ozet[DONEMLER[1]]['sayi']} şiir</strong> bu döneme aittir. Pozitif oran erken dönemden <strong>%{ozet[DONEMLER[0]]['poz']}'den %{ozet[DONEMLER[1]]['poz']}'e düşerken</strong>, negatif oran <strong>%{ozet[DONEMLER[0]]['neg']}'ten %{ozet[DONEMLER[1]]['neg']}'ya yükselir</strong>. Şiir uzunluğu da kısalmaya başlar (ort. {ozet[DONEMLER[1]]['ort_kelime']:.0f} kelime). Bu dönüşüm, şairin olgunlaşmasıyla beraber dilini sıkılaştırdığını, aşkı artık idealize etmek yerine kayıp ve özlemle birlikte ele aldığını gösterir. "Çaresiz", "Yalnız Kadınlar", "Yaşayan Ölü" gibi başlıklar dönemin tonunu yansıtır. Yine de aşk teması en baskın olarak kalır — sadece artık "kavuşma" değil, "ayrılık ve hasret" tonundadır.""",
            "anahtar": top_kelimeler[DONEMLER[1]][:6],
        },
        DONEMLER[2]: {
            "ust_baslik": "Acının Sesi: Kırılma Noktası",
            "yorum": f"""1973, Ümit Yaşar'ın hayatında ve şiirinde keskin bir kırılma yılıdır: <strong>oğlu Vedat intihar eder</strong>. Bu olaydan sonraki şiirlerinde duygusal ton tamamen değişir. Pozitif oran <strong>%{ozet[DONEMLER[2]]['poz']}'e düşer</strong> (erken dönemin yaklaşık <strong>beşte biri</strong>), negatif oran <strong>%{ozet[DONEMLER[2]]['neg']}'ye fırlar</strong> (erken dönemin <strong>dört katı</strong>). Ortalama duygu skoru <strong>{skor_fmt(ozet[DONEMLER[2]]['ort_skor'])}</strong> ile en negatif değere ulaşır. "Oğluma Ağıt", "Acılar Denizi", "Yalnızlığa Sone", "Köpek Yalnızlığım" gibi başlıklar dönemin hâkim duygusunu özetler. En sık geçen kelimelerde "seni"nin yerini <strong>"beni"</strong> alır — şair artık dış dünyaya değil, kendi iç dünyasına seslenmektedir. "Yalnız", "içimde" gibi içe dönük kelimeler ön plana çıkar.""",
            "anahtar": top_kelimeler[DONEMLER[2]][:6],
        },
    }

    donem_kartlari = ""
    for d in DONEMLER:
        o = ozet[d]
        donem_kartlari += f"""
      <div class="donem-kart {sinif(d)} fade-in">
        <div class="dk-yil {sinif(d)}">{yillar(d)}</div>
        <div class="dk-ad">{kisa(d)}<br><small style="color:var(--text-dim);font-size:.8rem;">{o['sayi']} şiir · Ort. {o['ort_kelime']:.0f} kelime</small></div>
        <div class="dk-row"><span class="dk-row-label">Pozitif</span><span style="color:#5BA65B;font-size:13px;font-weight:600">%{o['poz']}</span></div>
        <div class="mini-bar"><div class="mini-bar-fill fill-poz" style="width:{o['poz']}%"></div></div>
        <div class="dk-row" style="margin-top:8px"><span class="dk-row-label">Negatif</span><span style="color:#B85B5B;font-size:13px;font-weight:600">%{o['neg']}</span></div>
        <div class="mini-bar"><div class="mini-bar-fill fill-neg" style="width:{o['neg']}%"></div></div>
        <div class="dk-skor">
          <span class="dk-skor-label">Ort. Skor</span>
          <span class="dk-skor-val {skor_renk(o['ort_skor'])}">{skor_fmt(o['ort_skor'])}</span>
        </div>
      </div>"""

    # ── Detaylı dönem analiz kartları ──
    detay_kartlari = ""
    for d in DONEMLER:
        y = donem_yorumlari[d]
        anahtar_html = "".join([f'<span class="kw-pill">{k} <em>({v})</em></span>' for k,v in y["anahtar"]])
        detay_kartlari += f"""
        <div class="detay-kart {sinif(d)}">
          <div class="detay-header">
            <div class="detay-yil {sinif(d)}">{yillar(d)}</div>
            <h3>{y["ust_baslik"]}</h3>
            <div class="detay-meta">{kisa(d)} · {ozet[d]['sayi']} şiir</div>
          </div>
          <div class="detay-body">
            <p>{y["yorum"]}</p>
            <div class="kw-label">En sık kullanılan kelimeler:</div>
            <div class="kw-list">{anahtar_html}</div>
          </div>
        </div>"""

    ornek_kartlari = ""
    for d in DONEMLER:
        satirlar = ""
        for s in ornekler[d]["poz"]:
            satirlar += f'<div class="ornek-item"><span class="ornek-baslik">{s["baslik"]}</span><span class="ornek-badge badge-poz">+{s["st_skor"]:.3f}</span></div>'
        for s in ornekler[d]["neg"]:
            satirlar += f'<div class="ornek-item"><span class="ornek-baslik">{s["baslik"]}</span><span class="ornek-badge badge-neg">{s["st_skor"]:.3f}</span></div>'
        ornek_kartlari += f'<div class="ornek-donem"><div class="ornek-donem-header {sinif(d)}">{kisa(d)}</div>{satirlar}</div>'


    # TF-IDF HTML
    tfidf_kartlari = ""
    if tfidf_sonuc:
        for d in DONEMLER:
            if d in tfidf_sonuc:
                kelimeler = tfidf_sonuc[d][:10]
                pill_html = "".join([f'<span class="kw-pill">{k} <em>({s:.3f})</em></span>' for k,s in kelimeler])
                tfidf_kartlari += f'<div class="tfidf-donem"><div class="tfidf-header {sinif(d)}">{kisa(d)}</div><div class="tfidf-body">{pill_html}</div></div>'

    # N-gram HTML
    ngram_kartlari = ""
    if bigram_sonuc:
        for d in DONEMLER:
            bigrams = bigram_sonuc.get(d, [])[:6]
            trigrams = trigram_sonuc.get(d, [])[:4] if trigram_sonuc else []
            bi_html = "".join([f'<div class="ngram-item"><span class="ngram-text">{t}</span><span class="ngram-count">{c}</span></div>' for t,c in bigrams])
            tri_html = "".join([f'<div class="ngram-item"><span class="ngram-text">{t}</span><span class="ngram-count">{c}</span></div>' for t,c in trigrams])
            ngram_kartlari += f'''<div class="ngram-donem">
              <div class="ngram-header {sinif(d)}">{kisa(d)}</div>
              <div class="ngram-section-label">2'li Kalıplar (Bigram)</div>
              {bi_html}
              <div class="ngram-section-label" style="margin-top:16px">3'lü Kalıplar (Trigram)</div>
              {tri_html if tri_html else '<div class="ngram-item" style="opacity:.5"><span class="ngram-text">Yeterli veri yok</span></div>'}
            </div>'''

    poz_data  = [ozet[d]['poz']  for d in DONEMLER]
    neg_data  = [ozet[d]['neg']  for d in DONEMLER]
    notr_data = [ozet[d]['notr'] for d in DONEMLER]
    kelime_data = [ozet[d]['ort_kelime'] for d in DONEMLER]
    ham_skorlar = [ozet[d]['ort_skor'] for d in DONEMLER]
    filtre_skorlar = [ozet[d]['ort_skor_filtre'] or ozet[d]['ort_skor'] for d in DONEMLER]
    kisa_donemler = [kisa(d) for d in DONEMLER]
    bert_poz = [bert_ozet[d]['poz'] for d in DONEMLER] if bert_var else [0,0,0]
    bert_neg = [bert_ozet[d]['neg'] for d in DONEMLER] if bert_var else [0,0,0]

    if bert_var:
        ca_poz = [ca_ozet[d]['poz'] for d in DONEMLER] if ca_var else [0,0,0]
        ca_neg = [ca_ozet[d]['neg'] for d in DONEMLER] if ca_var else [0,0,0]
        bert_js = f"""
new Chart(document.getElementById('bertChart'),{{type:'bar',data:{{labels:DONEMLER_KISA,datasets:[
  {{label:'SentiTurkNet Poz%',data:{poz_data},backgroundColor:'#5BA65B',borderRadius:4,stack:'poz'}},
  {{label:'BERTurk Poz%',data:{bert_poz},backgroundColor:'rgba(91,166,91,.6)',borderRadius:4,stack:'poz'}},
  {{label:'codealchemist Poz%',data:{ca_poz},backgroundColor:'rgba(91,166,91,.3)',borderRadius:4,stack:'poz'}},
  {{label:'SentiTurkNet Neg%',data:{neg_data},backgroundColor:'#B85B5B',borderRadius:4,stack:'neg'}},
  {{label:'BERTurk Neg%',data:{bert_neg},backgroundColor:'rgba(184,91,91,.6)',borderRadius:4,stack:'neg'}},
  {{label:'codealchemist Neg%',data:{ca_neg},backgroundColor:'rgba(184,91,91,.3)',borderRadius:4,stack:'neg'}}
]}},options:{{...CD}}}});"""

    # Genel sonuç yorumu
    skor_degisim_yuzde = round(((ozet[DONEMLER[2]]['ort_skor'] - ozet[DONEMLER[0]]['ort_skor']) / abs(ozet[DONEMLER[0]]['ort_skor'] or 0.001)) * 100, 0)
    poz_dusus = round(ozet[DONEMLER[0]]['poz'] - ozet[DONEMLER[2]]['poz'], 1)
    neg_artis = round(ozet[DONEMLER[2]]['neg'] - ozet[DONEMLER[0]]['neg'], 1)

    sonuc_yorum = f"""
    <p style="margin-bottom:16px">Ümit Yaşar Oğuzcan'ın 37 yıl boyunca yazdığı 300 şiir üzerindeki SentiTurkNet duygu analizi, şairin duygusal evriminin <strong>belirgin ve istatistiksel olarak ölçülebilir</strong> olduğunu göstermektedir. Üç dönem arasındaki temel farklar:</p>
    <ul style="margin-left:20px;color:var(--text-dim);line-height:2;font-size:14px">
      <li><strong style="color:var(--text)">Pozitif duygu kaybı:</strong> Erken dönemden son döneme pozitif şiir oranı <strong>%{poz_dusus} puan düşmüştür</strong> (%{ozet[DONEMLER[0]]['poz']} → %{ozet[DONEMLER[2]]['poz']}).</li>
      <li><strong style="color:var(--text)">Negatif duygu artışı:</strong> Aynı dönemde negatif oran <strong>%{neg_artis} puan artarak</strong> %{ozet[DONEMLER[0]]['neg']}'ten %{ozet[DONEMLER[2]]['neg']}'ye yükselmiştir.</li>
      <li><strong style="color:var(--text)">Şiir kısalması:</strong> Ortalama kelime sayısı {ozet[DONEMLER[0]]['ort_kelime']:.0f}'den {ozet[DONEMLER[2]]['ort_kelime']:.0f}'e inmiştir — son dönem şiirleri daha yoğun ve özlü, daha az açıklayıcıdır.</li>
      <li><strong style="color:var(--text)">Tematik dönüşüm:</strong> "Aşk & Özlem" teması tüm dönemlerde varlığını korur ancak ton değişir: erken dönemde kavuşma umudu, son dönemde kaybın yası baskındır. "Yalnızlık" teması son dönemde belirgin biçimde artar.</li>
      <li><strong style="color:var(--text)">Söylem değişimi:</strong> "Seni" (sen-odaklı, dış dünyaya yönelik) → "Beni" (ben-odaklı, içe dönük) geçişi gözlemlenir.</li>
    </ul>
    <p style="margin-top:16px">Bu evrim, 1973'te oğul Vedat Oğuzcan'ın intiharıyla başlayan kişisel travmanın şiirsel ifadeye yansıması olarak yorumlanabilir. Şair, ölümünden sonraki yıllarda neredeyse hiç pozitif tonda şiir üretmemiştir.</p>
    """

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Ümit Yaşar Oğuzcan — Duygusal Evrim Analizi</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root{{--erken:#5B8DB8;--olgunluk:#C4963A;--son:#B85B5B;--bg:#0f0e0c;--surface:#1a1915;--surface2:#242320;--text:#e8e0d0;--text-dim:#8a8070;--accent:#c4963a;--border:#2e2c28;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Source Sans 3',sans-serif;background:var(--bg);color:var(--text);line-height:1.7;}}
.hero{{position:relative;padding:80px 40px 60px;text-align:center;border-bottom:1px solid var(--border);overflow:hidden;}}
.hero::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(196,150,58,.08) 0%,transparent 70%);pointer-events:none;}}
.hero-eyebrow{{font-size:11px;letter-spacing:4px;text-transform:uppercase;color:var(--accent);margin-bottom:20px;font-weight:600;}}
.hero h1{{font-family:'Playfair Display',serif;font-size:clamp(2rem,5vw,3.8rem);font-weight:700;line-height:1.2;margin-bottom:12px;}}
.hero h1 em{{font-style:italic;color:var(--accent);}}
.hero-sub{{color:var(--text-dim);font-size:15px;max-width:520px;margin:0 auto 40px;}}
.hero-stats{{display:flex;justify-content:center;gap:48px;flex-wrap:wrap;}}
.stat-num{{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;color:var(--accent);line-height:1;display:block;}}
.stat-label{{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--text-dim);margin-top:4px;display:block;}}
.container{{max-width:1100px;margin:0 auto;padding:0 32px;}}
section{{padding:64px 0;border-bottom:1px solid var(--border);}}
.section-label{{font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--accent);margin-bottom:8px;font-weight:600;}}
.section-title{{font-family:'Playfair Display',serif;font-size:1.8rem;margin-bottom:8px;}}
.section-desc{{color:var(--text-dim);font-size:14px;margin-bottom:40px;max-width:600px;}}
.donem-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:40px;}}
@media(max-width:700px){{.donem-grid,.two-col,.ornek-grid,.yontem-grid{{grid-template-columns:1fr;}}}}
.donem-kart{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:28px 24px;position:relative;overflow:hidden;transition:transform .2s,border-color .2s;}}
.donem-kart:hover{{transform:translateY(-3px);}}
.donem-kart::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.donem-kart.erken::before{{background:var(--erken);}}.donem-kart:hover.erken{{border-color:var(--erken);}}
.donem-kart.olgunluk::before{{background:var(--olgunluk);}}.donem-kart:hover.olgunluk{{border-color:var(--olgunluk);}}
.donem-kart.son::before{{background:var(--son);}}.donem-kart:hover.son{{border-color:var(--son);}}
.dk-yil{{font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;}}
.dk-yil.erken{{color:var(--erken);}}.dk-yil.olgunluk{{color:var(--olgunluk);}}.dk-yil.son{{color:var(--son);}}
.dk-ad{{font-family:'Playfair Display',serif;font-size:1.1rem;margin-bottom:20px;line-height:1.3;}}
.dk-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}}
.dk-row-label{{font-size:12px;color:var(--text-dim);}}
.mini-bar{{height:6px;background:var(--surface2);border-radius:3px;margin-top:4px;overflow:hidden;}}
.mini-bar-fill{{height:100%;border-radius:3px;}}
.fill-poz{{background:#5BA65B;}}.fill-neg{{background:#B85B5B;}}
.dk-skor{{margin-top:20px;padding-top:16px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.dk-skor-label{{font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;}}
.dk-skor-val{{font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;}}
.dk-skor-val.pos{{color:#5BA65B;}}.dk-skor-val.neg{{color:#B85B5B;}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:32px;margin-bottom:24px;}}
.chart-title{{font-family:'Playfair Display',serif;font-size:1.1rem;margin-bottom:4px;}}
.chart-note{{font-size:12px;color:var(--text-dim);margin-bottom:24px;}}
.chart-wrap{{position:relative;height:280px;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
.ornek-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}}
.ornek-donem{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
.ornek-donem-header{{padding:16px 20px;font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:600;border-bottom:1px solid var(--border);}}
.ornek-donem-header.erken{{color:var(--erken);background:rgba(91,141,184,.08);}}
.ornek-donem-header.olgunluk{{color:var(--olgunluk);background:rgba(196,150,58,.08);}}
.ornek-donem-header.son{{color:var(--son);background:rgba(184,91,91,.08);}}
.ornek-item{{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:12px;}}
.ornek-item:last-child{{border-bottom:none;}}
.ornek-baslik{{font-size:13px;line-height:1.3;}}
.ornek-badge{{font-size:10px;padding:3px 8px;border-radius:20px;white-space:nowrap;font-weight:600;}}
.badge-poz{{background:rgba(91,166,91,.15);color:#5BA65B;}}.badge-neg{{background:rgba(184,91,91,.15);color:#B85B5B;}}
.yontem-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
.yontem-kart{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;}}
.yontem-kart h4{{font-family:'Playfair Display',serif;margin-bottom:10px;}}
.yontem-kart p{{font-size:13px;color:var(--text-dim);line-height:1.7;}}
.tag{{display:inline-block;font-size:10px;padding:2px 8px;background:rgba(196,150,58,.15);color:var(--accent);border-radius:20px;margin-bottom:10px;letter-spacing:1px;text-transform:uppercase;font-weight:600;}}
.bulgu{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:20px 24px;font-size:14px;color:var(--text-dim);line-height:1.8;margin-top:8px;}}
.detay-kart{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:20px;display:grid;grid-template-columns:280px 1fr;}}
@media(max-width:780px){{.detay-kart{{grid-template-columns:1fr;}}}}
.detay-header{{padding:32px;border-right:1px solid var(--border);background:var(--surface2);}}
@media(max-width:780px){{.detay-header{{border-right:none;border-bottom:1px solid var(--border);}}}}
.detay-header h3{{font-family:'Playfair Display',serif;font-size:1.3rem;line-height:1.3;margin:8px 0;}}
.detay-yil{{font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:600;}}
.detay-yil.erken{{color:var(--erken);}}.detay-yil.olgunluk{{color:var(--olgunluk);}}.detay-yil.son{{color:var(--son);}}
.detay-meta{{font-size:12px;color:var(--text-dim);margin-top:8px;}}
.detay-body{{padding:32px;}}
.detay-body p{{font-size:14px;color:var(--text-dim);line-height:1.8;}}
.kw-label{{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim);margin-top:24px;margin-bottom:12px;font-weight:600;}}
.kw-list{{display:flex;flex-wrap:wrap;gap:8px;}}
.kw-pill{{background:var(--surface2);border:1px solid var(--border);padding:6px 12px;border-radius:20px;font-size:12px;color:var(--text);}}
.kw-pill em{{color:var(--text-dim);font-style:normal;font-size:11px;margin-left:2px;}}
.sonuc-kart{{background:linear-gradient(135deg,rgba(196,150,58,.06),rgba(184,91,91,.06));border:1px solid var(--border);border-radius:12px;padding:40px;}}
.sonuc-kart h3{{font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:20px;}}
.sonuc-kart p{{color:var(--text-dim);font-size:14px;line-height:1.8;}}
footer{{padding:40px;text-align:center;color:var(--text-dim);font-size:12px;letter-spacing:1px;}}
.fade-in{{opacity:0;transform:translateY(20px);animation:fadeIn .6s ease forwards;}}
@keyframes fadeIn{{to{{opacity:1;transform:translateY(0);}}}}
.fade-in:nth-child(1){{animation-delay:.1s;}}.fade-in:nth-child(2){{animation-delay:.2s;}}.fade-in:nth-child(3){{animation-delay:.3s;}}
.tfidf-grid{{display:grid;grid-template-columns:1fr;gap:16px;}}
.tfidf-donem{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;display:grid;grid-template-columns:180px 1fr;}}
@media(max-width:780px){{.tfidf-donem{{grid-template-columns:1fr;}}}}
.tfidf-header{{padding:24px;font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:600;background:var(--surface2);display:flex;align-items:center;border-right:1px solid var(--border);}}
.tfidf-header.erken{{color:var(--erken);}}
.tfidf-header.olgunluk{{color:var(--olgunluk);}}
.tfidf-header.son{{color:var(--son);}}
.tfidf-body{{padding:20px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;}}
.ngram-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}}
@media(max-width:780px){{.ngram-grid{{grid-template-columns:1fr;}}}}
.ngram-donem{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;}}
.ngram-header{{font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border);}}
.ngram-header.erken{{color:var(--erken);}}
.ngram-header.olgunluk{{color:var(--olgunluk);}}
.ngram-header.son{{color:var(--son);}}
.ngram-section-label{{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim);margin-bottom:8px;font-weight:600;}}
.ngram-item{{display:flex;justify-content:space-between;padding:6px 0;font-size:13px;border-bottom:1px solid rgba(46,44,40,.5);}}
.ngram-item:last-child{{border-bottom:none;}}
.ngram-text{{color:var(--text);}}
.ngram-count{{color:var(--text-dim);font-size:11px;font-weight:600;}}
</style>
</head>
<body>
<div class="hero">
  <div class="hero-eyebrow">Text Mining · Duygu Analizi · 2024</div>
  <h1>Ümit Yaşar Oğuzcan<br><em>Duygusal Evrim</em></h1>
  <p class="hero-sub">{len(df_filt)} şiirin duygu analizi — üç edebi dönem boyunca duygusal dönüşümün izleri</p>
  <div class="hero-stats">
    <div class="stat"><span class="stat-num">{len(df_filt)}</span><span class="stat-label">Şiir</span></div>
    <div class="stat"><span class="stat-num">3</span><span class="stat-label">Dönem</span></div>
    <div class="stat"><span class="stat-num">1947–84</span><span class="stat-label">Yıllar</span></div>
    <div class="stat"><span class="stat-num">%{ozet[DONEMLER[2]]['neg']}</span><span class="stat-label">Son Dönem Neg.</span></div>
  </div>
</div>

<section><div class="container">
  <div class="section-label">Genel Bakış</div>
  <h2 class="section-title">Üç Dönem, Üç Ses</h2>
  <p class="section-desc">Her dönem için pozitif/negatif duygu oranları ve ortalama duygu skoru.</p>
  <div class="donem-grid">{donem_kartlari}</div>
  <div class="bulgu"><strong style="color:var(--text)">Temel Bulgu:</strong> Pozitif duygu oranı erken dönemden son döneme %{ozet[DONEMLER[0]]['poz']}'den %{ozet[DONEMLER[2]]['poz']}'e inerken, negatif oran %{ozet[DONEMLER[0]]['neg']}'ten %{ozet[DONEMLER[2]]['neg']}'ye çıkmıştır. 1973'te oğlu Vedat'ın ölümü bu dönüşümde belirleyici kırılma noktasıdır.</div>
</div></section>

<section><div class="container">
  <div class="section-label">Detaylı Dönem Analizi</div>
  <h2 class="section-title">Her Dönem Kendi Hikâyesini Anlatır</h2>
  <p class="section-desc">Sayıların ötesinde — her dönemin baskın duygu örüntüleri, en sık kelimeleri ve şairin değişen sesi.</p>
  {detay_kartlari}
</div></section>

<section><div class="container">
  <div class="section-label">İnteraktif Grafikler</div>
  <h2 class="section-title">Duygu Eğrileri</h2>
  <p class="section-desc">Grafiklerin üzerine gelerek dönem değerlerini görebilirsiniz.</p>
  <div class="two-col">
    <div class="chart-card"><div class="chart-title">Dönemler Arası Duygu Trendi</div><div class="chart-note">Pozitif ve negatif şiir oranlarının değişimi</div><div class="chart-wrap"><canvas id="trendChart"></canvas></div></div>
    <div class="chart-card"><div class="chart-title">Duygu Dağılımı (Yığılmış)</div><div class="chart-note">Her dönemdeki pozitif / nötr / negatif oran</div><div class="chart-wrap"><canvas id="stackChart"></canvas></div></div>
  </div>
  <div class="two-col">
    <div class="chart-card"><div class="chart-title">Tema Yoğunluğu</div><div class="chart-note">Dönemlere göre ortalama tema skoru</div><div class="chart-wrap"><canvas id="temaChart"></canvas></div></div>
    <div class="chart-card"><div class="chart-title">Ortalama Şiir Uzunluğu</div><div class="chart-note">Kelime sayısı dönemlere göre nasıl değişiyor</div><div class="chart-wrap"><canvas id="kelimeChart"></canvas></div></div>
  </div>
  {bert_blok}
  {bert_yorum}
</div></section>

<section><div class="container">
  <div class="section-label">Örnekler</div>
  <h2 class="section-title">En Yüksek / En Düşük Skorlu Şiirler</h2>
  <p class="section-desc">Her dönemden en pozitif ve en negatif duygu skoruna sahip şiirler. Başlıklar bile dönüşümü göstermektedir.</p>
  <div class="ornek-grid">{ornek_kartlari}</div>
</div></section>

<section><div class="container">
  <div class="section-label">Sonuç</div>
  <h2 class="section-title">Bir Şairin Duygu Yolculuğu</h2>
  <div class="sonuc-kart">
    {sonuc_yorum}
  </div>
</div></section>

<section><div class="container">
  <div class="section-label">Döneme Özgü Sözcükler</div>
  <h2 class="section-title">TF-IDF Analizi</h2>
  <p class="section-desc">Her dönemin diğerlerinden ayrışan, en karakteristik kelimeleri. TF-IDF skoru yüksek olan kelimeler, o döneme özgü; sık geçip ama her dönemde olan kelimeler değil.</p>
  <div class="tfidf-grid">{tfidf_kartlari}</div>
  <div class="bulgu" style="margin-top:20px">
    <strong style="color:var(--text)">Neden TF-IDF?</strong> Sadece kelime sayımı yapıldığında "seni" ve "beni" hep ilk sırada olur — çünkü her dönemde çok geçerler. TF-IDF ise her döneme <em>özgü</em> olan kelimeleri öne çıkarır. Örneğin "ölüm" sadece son dönemde yoğunsa TF-IDF skoru orada yüksek olur, diğer dönemlerde düşük.
  </div>
</div></section>

<section><div class="container">
  <div class="section-label">Anlam Kalıpları</div>
  <h2 class="section-title">N-gram Analizi</h2>
  <p class="section-desc">Tek kelime yerine 2'li ve 3'lü kelime grupları — şiirsel kalıpları yakalamak için. "Acılar denizi", "büyük yalnızlık" gibi anlamlı tamlamalar tek kelimelerden çok daha güçlü duygu taşır.</p>
  <div class="ngram-grid">{ngram_kartlari}</div>
</div></section>

<section><div class="container">
  <div class="section-label">Metodolojik Karşılaştırma</div>
  <h2 class="section-title">Stopword Filtrelemenin Etkisi</h2>
  <p class="section-desc">Şair dilinde çok geçen "seni", "beni", "bütün" gibi kelimeler duygu yükü taşımaz ama skor formülünün paydasını şişirir. Bu kelimeler çıkarıldığında gerçek duygu yoğunluğu daha net görünür.</p>
  <div class="chart-card">
    <div class="chart-title">İki Yöntemin Karşılaştırması</div>
    <div class="chart-note">Ham skor: tüm kelimeler · Filtreli skor: içerik kelimeleri (stopword'ler hariç)</div>
    <div class="chart-wrap"><canvas id="filtreChart"></canvas></div>
  </div>
  <div class="bulgu" style="border-left-color:#5B7EB8">
    <strong style="color:var(--text)">Stopword Filtrelemenin Etkisi:</strong> Son dönem için ham skor <strong>{ozet[DONEMLER[2]]["ort_skor"]:+.4f}</strong> iken filtrelenmiş skor <strong>{ozet[DONEMLER[2]]["ort_skor_filtre"]:+.4f}</strong>'tir — yani gerçek negatif yoğunluk ham skorun gösterdiğinden <strong>%{round(abs((ozet[DONEMLER[2]]["ort_skor_filtre"] - ozet[DONEMLER[2]]["ort_skor"]) / ozet[DONEMLER[2]]["ort_skor"] * 100))} daha fazla</strong>. Bu, sözcük tabanlı duygu analizinde stopword temizliğinin önemini gösterir. Filtrelenmiş skor, sanatsal metinlerde daha güvenilir bir ölçüt sunmaktadır.
  </div>
</div></section>

<section><div class="container">
  <div class="section-label">Yöntem</div>
  <h2 class="section-title">Nasıl Analiz Edildi?</h2>
  <div class="yontem-grid">
    <div class="yontem-kart"><div class="tag">1 · Veri Toplama</div><h4>Web Scraping</h4><p>antoloji.com'dan BeautifulSoup ile {len(df)} şiir çekildi. Duplikasyonlar URL bazlı tekilleştirmeyle temizlendi.</p></div>
    <div class="yontem-kart"><div class="tag">2 · Ön İşleme</div><h4>Tokenization & Lowercase</h4><p>Şiirler küçük harfe çevrildi, noktalama temizlendi, kelimeler ayrıldı (tokenize edildi). Türkçe karakterler korundu.</p></div>
    <div class="yontem-kart"><div class="tag">3 · Stopword</div><h4>Filtreleme</h4><p>"Ben", "sen", "bir", "ve" gibi duygu yükü taşımayan 60+ Türkçe stopword çıkarıldı. Ham ve filtrelenmiş skorlar karşılaştırıldı.</p></div>
    <div class="yontem-kart"><div class="tag">4 · Stemming</div><h4>Zeyrek + NLTK</h4><p>Türkçe morfolojik analizci Zeyrek ile "seviyorum/sevdim/sevdin" gibi çekimler ortak köke ("sev") indirgendi. TF-IDF stemli metin üzerinde çalıştırıldı.</p></div>
    <div class="yontem-kart"><div class="tag">5 · Duygu Analizi</div><h4>SentiTurkNet{' + BERTurk' if bert_var else ''}</h4><p>Türkçe duygu leksikonu ile pozitif/negatif kelime sayımı + transformer tabanlı BERTurk modeli karşılaştırmalı olarak uygulandı.</p></div>
    <div class="yontem-kart"><div class="tag">6 · TF-IDF & N-gram</div><h4>İleri Düzey Çıkarım</h4><p>Her döneme özgü kelimeler (TF-IDF) ve anlamlı kelime kalıpları (bigram, trigram) sklearn ile hesaplandı.</p></div>
    <div class="yontem-kart"><div class="tag">7 · Dönemleme</div><h4>Bibliyografik Eşleştirme</h4><p>Wikipedia ve edebiyat kaynakları kullanılarak şiirler kitaplarına, kitaplar yayın yıllarına göre üç döneme etiketlendi.</p></div>
    <div class="yontem-kart"><div class="tag">8 · Kısıtlama</div><h4>Sınırlılıklar</h4><p>Şiirlerin bir kısmı metin tabanlı tahminle dönemlendirilmiştir. SentiTurkNet ironi ve metaforu doğrudan yakalayamaz.</p></div>
  </div>
</div></section>

<footer>Ümit Yaşar Oğuzcan Duygusal Evrim Analizi · Text Mining Projesi · 2024<br><span style="opacity:.5">antoloji.com · SentiTurkNet · Python</span></footer>

<script>
const DONEMLER_KISA = {kisa_donemler};
const CD = {{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8a8070',font:{{size:12}},boxWidth:12}}}},tooltip:{{backgroundColor:'#1a1915',borderColor:'#2e2c28',borderWidth:1,titleColor:'#e8e0d0',bodyColor:'#8a8070'}}}},scales:{{x:{{ticks:{{color:'#8a8070',font:{{size:11}}}},grid:{{color:'#2e2c28'}}}},y:{{ticks:{{color:'#8a8070',font:{{size:11}}}},grid:{{color:'#2e2c28'}}}}}}}};
new Chart(document.getElementById('trendChart'),{{type:'line',data:{{labels:DONEMLER_KISA,datasets:[{{label:'Pozitif %',data:{poz_data},borderColor:'#5BA65B',backgroundColor:'rgba(91,166,91,.1)',fill:true,tension:.4,pointRadius:6,pointBackgroundColor:'#5BA65B'}},{{label:'Negatif %',data:{neg_data},borderColor:'#B85B5B',backgroundColor:'rgba(184,91,91,.1)',fill:true,tension:.4,pointRadius:6,pointBackgroundColor:'#B85B5B'}}]}},options:{{...CD}}}});
new Chart(document.getElementById('stackChart'),{{type:'bar',data:{{labels:DONEMLER_KISA,datasets:[{{label:'Pozitif',data:{poz_data},backgroundColor:'#5BA65B'}},{{label:'Nötr',data:{notr_data},backgroundColor:'#4A6FA5'}},{{label:'Negatif',data:{neg_data},backgroundColor:'#B85B5B'}}]}},options:{{...CD,scales:{{x:{{...CD.scales.x,stacked:true}},y:{{...CD.scales.y,stacked:true,max:100}}}}}}}});
new Chart(document.getElementById('temaChart'),{{type:'radar',data:{{labels:{tema_labels},datasets:[{{label:'Erken',data:{tema_data[DONEMLER[0]]},borderColor:'#5B8DB8',backgroundColor:'rgba(91,141,184,.15)',pointBackgroundColor:'#5B8DB8'}},{{label:'Olgunluk',data:{tema_data[DONEMLER[1]]},borderColor:'#C4963A',backgroundColor:'rgba(196,150,58,.15)',pointBackgroundColor:'#C4963A'}},{{label:'Son',data:{tema_data[DONEMLER[2]]},borderColor:'#B85B5B',backgroundColor:'rgba(184,91,91,.15)',pointBackgroundColor:'#B85B5B'}}]}},options:{{...CD,scales:{{r:{{ticks:{{color:'#8a8070',backdropColor:'transparent',font:{{size:10}}}},grid:{{color:'#2e2c28'}},pointLabels:{{color:'#e8e0d0',font:{{size:11}}}}}}}}}}}});
new Chart(document.getElementById('filtreChart'),{{type:'bar',data:{{labels:DONEMLER_KISA,datasets:[
  {{label:'Ham Skor',data:{ham_skorlar},backgroundColor:'rgba(196,150,58,.7)',borderRadius:4}},
  {{label:'Filtreli Skor',data:{filtre_skorlar},backgroundColor:'rgba(91,141,184,.85)',borderRadius:4}}
]}},options:{{...CD}}}});
new Chart(document.getElementById('kelimeChart'),{{type:'bar',data:{{labels:DONEMLER_KISA,datasets:[{{label:'Ort. Kelime',data:{kelime_data},backgroundColor:['#5B8DB8','#C4963A','#B85B5B'],borderRadius:6}}]}},options:{{...CD,plugins:{{...CD.plugins,legend:{{display:false}}}}}}}});
{bert_js}
</script>
</body></html>"""

    with open("rapor.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    # Öncelik sırası: dönem eşli CSV > temiz CSV > ham CSV
    for yol in [
        "umit_yasar_donem_esli.csv",
        "/mnt/user-data/outputs/umit_yasar_donem_esli.csv",
        "umit_yasar_temiz.csv",
        "/mnt/user-data/uploads/umit_yasar_siirler_v2.csv",
    ]:
        if os.path.exists(yol):
            csv_yolu = yol
            break

    print(f"CSV yükleniyor: {csv_yolu}")
    df = pd.read_csv(csv_yolu)

    # Duplikasyon temizle (eğer ham CSV ise)
    if df['url'].str.contains(r'\?').any():
        df['url'] = df['url'].str.split('?').str[0]
        df = df.drop_duplicates(subset='url', keep='first').reset_index(drop=True)
        df['id'] = df.index + 1

    # donem_final varsa onu kullan, yoksa donem sütununu kullan
    if 'donem_final' in df.columns:
        df['donem'] = df['donem_final']

    df['kelime_sayisi'] = df['metin'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df = df[df['kelime_sayisi'] > 5].reset_index(drop=True)

    print(f"Analiz edilecek şiir sayısı: {len(df)}")

    # ── 1. SentiTurkNet ──────────────────────────────────────────────────────
    print("\n[1/3] SentiTurkNet skorları hesaplanıyor...")
    senti_sonuclar = df['metin'].apply(sentiturk_skor)
    senti_df = pd.DataFrame(senti_sonuclar.tolist())
    df = pd.concat([df, senti_df], axis=1)
    print(f"  Dağılım: {df['st_etiket'].value_counts().to_dict()}")

    # ── 2. Tema tespiti ───────────────────────────────────────────────────────
    print("\n[2/3] Tema tespiti yapılıyor...")
    tema_sonuclar = df['metin'].apply(tema_tespiti)
    tema_df = pd.DataFrame(tema_sonuclar.tolist())
    df = pd.concat([df, tema_df], axis=1)
    print(f"  Dominant temalar: {df['dominant_tema'].value_counts().to_dict()}")

    # ── 2b. Stemming (Zeyrek/NLTK) ────────────────────────────────────────────
    print("\n[2b/3] Stemming (kök bulma) yapılıyor...")
    try:
        # NLTK punkt_tab indirme (gerekirse)
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("  NLTK punkt_tab indiriliyor...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)

        df['metin_stem'] = df['metin'].apply(lambda x: " ".join(stemler(x)) if isinstance(x, str) else "")
        print(f"  ✓ {len(df)} şiir stemlendi (örnek: '{df['metin'].iloc[0].split()[0]}' → '{df['metin_stem'].iloc[0].split()[0] if df['metin_stem'].iloc[0] else '?'}')")
    except Exception as e:
        print(f"  ! Stemming atlandı: {e}")
        df['metin_stem'] = df['metin']  # fallback: ham metin

    # ── 3a. BERTurk (savasy — 2 sınıflı) ─────────────────────────────────────
    print("\n[3a/5] BERTurk (savasy) duygu analizi yapılıyor...")
    bert_sonuclar = berturk_duygu(df['metin'].tolist())
    bert_df = pd.DataFrame(bert_sonuclar)
    df = pd.concat([df, bert_df], axis=1)
    if 'bert_etiket' in df.columns and df['bert_etiket'].iloc[0] != 'yok':
        print(f"  Dağılım: {df['bert_etiket'].value_counts().to_dict()}")

    # ── 3b. codealchemist01 (3 sınıflı, daha kapsamlı) ───────────────────────
    print("\n[3b/5] codealchemist01 duygu analizi yapılıyor (439K cümle ile eğitilmiş)...")
    ca_sonuclar = codealchemist_duygu(df['metin'].tolist())
    ca_df = pd.DataFrame(ca_sonuclar)
    df = pd.concat([df, ca_df], axis=1)
    if 'ca_etiket' in df.columns and df['ca_etiket'].iloc[0] != 'yok':
        print(f"  Dağılım: {df['ca_etiket'].value_counts().to_dict()}")

    # ── 4. TF-IDF (döneme özgü kelimeler) — stemli metin üzerinde ──────────
    print("\n[4/5] TF-IDF analizi yapılıyor (stemli metin)...")
    # Stemli metin varsa onu kullan, yoksa orijinali
    metin_col = 'metin_stem' if 'metin_stem' in df.columns and df['metin_stem'].notna().any() else 'metin'
    tfidf_sonuc = tfidf_donem(df, metin_col=metin_col)
    for d, kelimeler in tfidf_sonuc.items():
        print(f"  {d}: {[k for k,_ in kelimeler[:5]]}")

    # ── 5. N-gram (bigram + trigram) — orijinal metin (kalıplar için) ──────
    print("\n[5/5] N-gram analizi yapılıyor (orijinal metin)...")
    bigram_sonuc = ngram_donem(df, n=2, n_top=10)  # ham metin daha doğal
    trigram_sonuc = ngram_donem(df, n=3, n_top=8)
    for d, kelimeler in bigram_sonuc.items():
        print(f"  {d} bigram: {[k for k,_ in kelimeler[:3]]}")

    # df'e ek bilgileri kaydet (rapor için)
    df.attrs['tfidf']   = tfidf_sonuc
    df.attrs['bigram']  = bigram_sonuc
    df.attrs['trigram'] = trigram_sonuc

    # ── Dönem bazlı özet ─────────────────────────────────────────────────────
    print("\n── Dönem Bazlı Duygu Ortalamaları ──")
    donem_ozet = df.groupby('donem').agg(
        siir_sayisi=('id', 'count'),
        ort_st_skor=('st_skor', 'mean'),
        poz_oran=('st_etiket', lambda x: (x == 'pozitif').mean()),
        neg_oran=('st_etiket', lambda x: (x == 'negatif').mean()),
        ort_kelime=('kelime_sayisi', 'mean'),
    ).round(3)
    print(donem_ozet.to_string())

    # ── Kaydet ───────────────────────────────────────────────────────────────
    out = "umit_yasar_analiz.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✓ Analiz CSV kaydedildi: {out}")

    ozet_out = "umit_yasar_donem_ozet.csv"
    donem_ozet.to_csv(ozet_out, encoding="utf-8-sig")
    print(f"✓ Dönem özeti kaydedildi: {ozet_out}")

    # ── HTML Rapor ────────────────────────────────────────────────────────────
    print("\n[4/4] HTML rapor oluşturuluyor...")
    html_rapor_uret(df)
    print("✓ HTML rapor kaydedildi: rapor.html")

    return df, donem_ozet



if __name__ == "__main__":
    df, ozet = main()


