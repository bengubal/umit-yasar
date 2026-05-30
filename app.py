from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import os
from collections import Counter

app = Flask(__name__)

POZITIF_KELIMELER = {
    "sevgi", "aşk", "mutlu", "güzel", "umut", "sevinç", "neşe", "huzur",
    "sevmek", "seviyorum", "güzellik", "iyilik", "şefkat", "sevi", "dost",
    "dostluk", "gülmek", "gülüş", "bahar", "çiçek", "ışık", "aydınlık",
    "özgür", "özgürlük", "barış", "şen", "tatlı", "canım", "sevgili",
    "harika", "muhteşem", "mükemmel", "beraber", "birlikte", "kavuşmak",
    "hasret", "özlem", "beklemek", "gelmek", "bulunmak", "yaşamak",
    "gülümse", "sıcak", "yumuşak", "nazik", "içten", "samimi", "güven",
    "inanmak", "ümit", "umutlu", "parlak", "aydın", "ferah", "rahat",
    "şükran", "minnet", "saygı", "neşeli", "canlı",
    "coşku", "heyecan", "tutku", "arzu", "istek", "özgün", "derin",
    "anlam", "değer", "kıymet", "önem", "hatıra", "anı",
}

NEGATIF_KELIMELER = {
    "acı", "ölüm", "ölmek", "yalnız", "yalnızlık", "karanlık", "ağlamak",
    "gözyaşı", "hüzün", "keder", "dert", "yas", "üzüntü", "ayrılık",
    "ayrılmak", "gitmek", "kayıp", "yitirmek", "boşluk",
    "çaresiz", "çaresizlik", "umutsuz", "umutsuzluk", "lanet", "bitmek",
    "tükenmek", "yorgun", "bitkin", "acımak", "sızı", "ağrı", "yara",
    "kan", "ağıt", "mezar", "ölü", "ceset", "toprak", "son",
    "mahkum", "hapis", "zindan", "karanlıklar", "gece",
    "soğuk", "donmak", "kasvet", "kabus", "korku", "dehşet",
    "felaket", "yıkım", "çöküş", "düşmek", "batmak",
    "sönmek", "solmak", "sararmak", "solgun",
    "inlemek", "sızlamak", "yanmak",
    "ihanet", "yalan", "aldatmak", "terk", "bırakmak",
    "unutmak", "yokluk", "hiçlik", "anlamsız",
    "beyhude", "boşuna", "faydasız", "zavallı", "biçare",
}

GUCLU_POZITIF = {"aşk", "sevgi", "mutluluk", "sevinç", "huzur", "umut", "kavuşmak"}
GUCLU_NEGATIF = {"ölüm", "acı", "yalnızlık", "ayrılık", "yas", "çaresiz", "umutsuz", "mezar"}

STOPWORDS = {
    "bir","ve","bu","ben","sen","o","biz","siz","onlar","de","da","ki","mı","mi","mu","mü",
    "ile","için","gibi","ama","ya","ne","var","yok","olur","olan","olarak","kadar","sonra",
    "önce","her","hiç","çok","az","hep","biraz","artık","yine","sadece","bile","daha","en",
    "şu","şimdi","ise","değil","vardı","nasıl","seni","sana","senin","beni","bana",
    "benim","bizi","bize","bizim","sizi","size","sizin","onu","ona","onun",
    "şey","gün","yıl","kez","kere","tüm","bütün","kendi",
    "evet","hayır","tabi","belki","mutlaka","kesin","yani",
}

TEMA_SOZLUK = {
    "ask_ozlem":    {"kelimeler": ["aşk","sevgi","sevmek","özlem","hasret","sevi","sevgili","kalp","seviyorum"], "label": "Aşk & Özlem"},
    "olum_aci":     {"kelimeler": ["ölüm","ölmek","acı","yas","mezar","ceset","ağıt","öldü","can"], "label": "Ölüm & Acı"},
    "yalnizlik":    {"kelimeler": ["yalnız","yalnızlık","kimsesiz","yapayalnız","tek","issız"], "label": "Yalnızlık"},
    "umut_kavusma": {"kelimeler": ["umut","ümit","kavuşmak","beklemek","gelecek","yeniden","yarın"], "label": "Umut & Kavuşma"},
    "doga":         {"kelimeler": ["bahar","güz","sonbahar","yağmur","deniz","rüzgar","gece","gün","güneş","mevsim"], "label": "Doğa"},
    "toplum":       {"kelimeler": ["halk","millet","ülke","vatan","savaş","insan","dünya","tarih","memleket"], "label": "Toplum & Vatan"},
}

DONEM_ACIKLAMALAR = {
    "Erken Dönem (1947–1962)": "Aşk ve romantizmin baskın olduğu, İstanbul ve deniz imgelerinin öne çıktığı ilk dönem.",
    "Olgunluk Dönemi (1963–1972)": "Üretkenliğin zirve yaptığı, hem aşk hem melankolinin bir arada bulunduğu olgunluk dönemi.",
    "Son Dönem (1973–1984)": "1973'te oğlu Vedat'ın kaybının ardından yalnızlık, acı ve ölüm temalarının baskın olduğu son dönem.",
}

df = None
tfidf_vectorizer = None
tfidf_matrix = None


def normalize_donem(s):
    if not isinstance(s, str):
        return "Belirsiz"
    if "Erken" in s:
        return "Erken Dönem (1947–1962)"
    if "Olgunluk" in s:
        return "Olgunluk Dönemi (1963–1972)"
    if "Son" in s:
        return "Son Dönem (1973–1984)"
    return s


def load_data():
    global df, tfidf_vectorizer, tfidf_matrix
    csv_yol = None
    for yol in ["umit_yasar_donem_esli.csv", "umit_yasar_siirler_v2.csv", "umit_yasar_siirler.csv"]:
        if os.path.exists(yol):
            csv_yol = yol
            break
    if not csv_yol:
        print("CSV bulunamadı!")
        return

    _df = None
    for enc in ["latin-1", "utf-8-sig", "utf-8", "cp1252"]:
        try:
            _df = pd.read_csv(csv_yol, encoding=enc)
            if "baslik" in _df.columns:
                break
        except Exception:
            continue

    if _df is None:
        print("CSV okunamadı!")
        return

    for col in ["donem", "donem_final"]:
        if col in _df.columns:
            _df[col] = _df[col].apply(normalize_donem)

    _df["kelime_sayisi"] = _df["metin"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    _df = _df[_df["kelime_sayisi"] > 5].reset_index(drop=True)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=list(STOPWORDS),
            token_pattern=r"(?u)\b[a-zçğıöşü]{2,}\b",
            lowercase=True,
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(_df["metin"].fillna("").tolist())
        print(f"TF-IDF matrisi hazır: {tfidf_matrix.shape}")
    except ImportError:
        print("sklearn yok, keyword tabanlı öneri kullanılacak.")

    df = _df
    print(f"Veri yüklendi: {len(df)} şiir")


def sentiturk_skor(metin: str) -> dict:
    if not isinstance(metin, str) or not metin.strip():
        return {"st_pozitif_sayi": 0, "st_negatif_sayi": 0, "st_skor": 0.0,
                "st_etiket": "nötr", "st_skor_filtre": 0.0, "st_etiket_filtre": "nötr",
                "icerik_kelime_sayisi": 0}

    kelimeler = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", metin.lower()).split()
    poz, neg = 0.0, 0.0
    for k in kelimeler:
        if k in GUCLU_POZITIF:         poz += 2.0
        elif k in POZITIF_KELIMELER:   poz += 1.0
        if k in GUCLU_NEGATIF:         neg += 2.0
        elif k in NEGATIF_KELIMELER:   neg += 1.0

    toplam = len(kelimeler) or 1
    skor = (poz - neg) / toplam

    icerik = [k for k in kelimeler if k not in STOPWORDS and len(k) > 2]
    toplam_icerik = len(icerik) or 1
    skor_filtre = (poz - neg) / toplam_icerik

    def etiketle(s):
        if s > 0.04:  return "pozitif"
        if s < -0.04: return "negatif"
        return "nötr"

    return {
        "st_pozitif_sayi": int(poz),
        "st_negatif_sayi": int(neg),
        "st_skor": round(skor, 4),
        "st_etiket": etiketle(skor * 2),
        "st_skor_filtre": round(skor_filtre, 4),
        "st_etiket_filtre": etiketle(skor_filtre),
        "icerik_kelime_sayisi": toplam_icerik,
    }


def tema_tespiti(metin: str) -> dict:
    if not isinstance(metin, str):
        return {"temalar": {}, "dominant": "genel", "dominant_label": "Genel"}
    m = metin.lower()
    sonuc = {}
    for key, val in TEMA_SOZLUK.items():
        sonuc[key] = sum(1 for k in val["kelimeler"] if k in m)
    dominant = max(sonuc, key=sonuc.get) if sonuc else None
    if dominant and sonuc[dominant] > 0:
        dominant_label = TEMA_SOZLUK[dominant]["label"]
    else:
        dominant, dominant_label = "genel", "Genel"
    return {"temalar": sonuc, "dominant": dominant, "dominant_label": dominant_label}


def tahmini_donem(skor_data: dict, tema_data: dict, kelime_sayisi: int) -> dict:
    skorlar = {
        "Erken Dönem (1947–1962)": 0,
        "Olgunluk Dönemi (1963–1972)": 0,
        "Son Dönem (1973–1984)": 0,
    }

    skor = skor_data.get("st_skor_filtre", skor_data.get("st_skor", 0))
    temalar = tema_data.get("temalar", {})

    if skor > 0.05:
        skorlar["Erken Dönem (1947–1962)"] += 4
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    elif skor > 0.01:
        skorlar["Erken Dönem (1947–1962)"] += 2
        skorlar["Olgunluk Dönemi (1963–1972)"] += 2
    elif skor < -0.06:
        skorlar["Son Dönem (1973–1984)"] += 4
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    elif skor < -0.01:
        skorlar["Son Dönem (1973–1984)"] += 2
        skorlar["Olgunluk Dönemi (1963–1972)"] += 2
    else:
        skorlar["Olgunluk Dönemi (1963–1972)"] += 3

    if temalar.get("yalnizlik", 0) >= 1:
        skorlar["Son Dönem (1973–1984)"] += 3
    if temalar.get("olum_aci", 0) >= 1:
        skorlar["Son Dönem (1973–1984)"] += 2
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    if temalar.get("ask_ozlem", 0) >= 3:
        skorlar["Erken Dönem (1947–1962)"] += 2
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    elif temalar.get("ask_ozlem", 0) >= 1:
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    if temalar.get("doga", 0) >= 2:
        skorlar["Erken Dönem (1947–1962)"] += 1
    if temalar.get("umut_kavusma", 0) >= 1:
        skorlar["Erken Dönem (1947–1962)"] += 1
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1
    if temalar.get("toplum", 0) >= 1:
        skorlar["Olgunluk Dönemi (1963–1972)"] += 1

    if kelime_sayisi > 100:
        skorlar["Erken Dönem (1947–1962)"] += 1
    elif kelime_sayisi < 40:
        skorlar["Son Dönem (1973–1984)"] += 1

    toplam = sum(skorlar.values()) or 1
    olasiliklar = {k: round(v / toplam * 100, 1) for k, v in skorlar.items()}
    tahmin = max(skorlar, key=skorlar.get)

    return {
        "tahmin": tahmin,
        "olasiliklar": olasiliklar,
        "aciklama": DONEM_ACIKLAMALAR.get(tahmin, ""),
    }


def _metin_ozet(metin, n=180):
    if not isinstance(metin, str):
        return ""
    return (metin[:n] + "...") if len(metin) > n else metin


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/oneri", methods=["POST"])
def oneri():
    data = request.get_json() or {}
    sorgu = data.get("sorgu", "").strip()
    tema_filtre = data.get("tema", "").strip()
    donem_filtre = data.get("donem", "").strip()
    limit = min(int(data.get("limit", 8)), 20)

    if df is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    calisma_df = df.copy()
    if donem_filtre and donem_filtre != "hepsi":
        calisma_df = calisma_df[calisma_df["donem_final"] == donem_filtre]

    sonuclar = []

    if sorgu and tfidf_vectorizer is not None and tfidf_matrix is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        # get positions of calisma_df rows in original df
        idx_map = list(calisma_df.index)
        sub_matrix = tfidf_matrix[idx_map]
        sorgu_vec = tfidf_vectorizer.transform([sorgu.lower()])
        benzerlikler = cosine_similarity(sorgu_vec, sub_matrix).flatten()
        sirali = benzerlikler.argsort()[::-1]
        for i in sirali[:limit]:
            if benzerlikler[i] < 0.005:
                break
            row = calisma_df.iloc[i]
            sonuclar.append({
                "baslik": row.get("baslik", ""),
                "donem": row.get("donem_final", ""),
                "kitap": row.get("kitap_adi", "") if not pd.isna(row.get("kitap_adi", "")) else "",
                "skor": round(float(benzerlikler[i]), 3),
                "metin_ozet": _metin_ozet(row.get("metin", "")),
                "url": row.get("url", ""),
            })
    elif tema_filtre and tema_filtre in TEMA_SOZLUK:
        tema_kels = TEMA_SOZLUK[tema_filtre]["kelimeler"]
        def tema_skor(metin):
            if not isinstance(metin, str): return 0
            m = metin.lower()
            return sum(1 for k in tema_kels if k in m)
        calisma_df = calisma_df.copy()
        calisma_df["_ts"] = calisma_df["metin"].apply(tema_skor)
        calisma_df = calisma_df[calisma_df["_ts"] > 0].nlargest(limit, "_ts")
        for _, row in calisma_df.iterrows():
            sonuclar.append({
                "baslik": row.get("baslik", ""),
                "donem": row.get("donem_final", ""),
                "kitap": row.get("kitap_adi", "") if not pd.isna(row.get("kitap_adi", "")) else "",
                "skor": round(float(row["_ts"]) / len(tema_kels), 3),
                "metin_ozet": _metin_ozet(row.get("metin", "")),
                "url": row.get("url", ""),
            })
    else:
        ornekler = calisma_df.sample(min(limit, len(calisma_df)), random_state=42)
        for _, row in ornekler.iterrows():
            sonuclar.append({
                "baslik": row.get("baslik", ""),
                "donem": row.get("donem_final", ""),
                "kitap": row.get("kitap_adi", "") if not pd.isna(row.get("kitap_adi", "")) else "",
                "skor": 0,
                "metin_ozet": _metin_ozet(row.get("metin", "")),
                "url": row.get("url", ""),
            })

    return jsonify({"siirler": sonuclar, "toplam": len(sonuclar)})


@app.route("/api/analiz", methods=["POST"])
def analiz():
    data = request.get_json() or {}
    metin = data.get("metin", "").strip()
    if not metin:
        return jsonify({"error": "Şiir metni gerekli"}), 400

    kelime_sayisi = len(metin.split())
    skor_data = sentiturk_skor(metin)
    tema_data = tema_tespiti(metin)
    donem_data = tahmini_donem(skor_data, tema_data, kelime_sayisi)

    kelimeler = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", metin.lower()).split()
    icerik = [k for k in kelimeler if k not in STOPWORDS and len(k) > 2]
    top_kelimeler = Counter(icerik).most_common(10)

    return jsonify({
        "duygu": skor_data,
        "temalar": tema_data,
        "donem": donem_data,
        "kelime_sayisi": kelime_sayisi,
        "top_kelimeler": top_kelimeler,
    })


@app.route("/api/rapor", methods=["POST"])
def rapor():
    data = request.get_json() or {}
    metin = data.get("metin", "").strip()
    baslik = data.get("baslik", "Başlıksız").strip() or "Başlıksız"
    if not metin:
        return jsonify({"error": "Şiir metni gerekli"}), 400

    kelime_sayisi = len(metin.split())
    satir_sayisi = len([s for s in metin.split("\n") if s.strip()])
    skor_data = sentiturk_skor(metin)
    tema_data = tema_tespiti(metin)
    donem_data = tahmini_donem(skor_data, tema_data, kelime_sayisi)

    kelimeler = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", metin.lower()).split()
    icerik = [k for k in kelimeler if k not in STOPWORDS and len(k) > 2]
    top_kelimeler = Counter(icerik).most_common(15)

    bigramlar = [f"{icerik[i]} {icerik[i+1]}" for i in range(len(icerik) - 1)]
    top_bigramlar = Counter(bigramlar).most_common(8)

    benzer_siirler = []
    if df is not None and tfidf_vectorizer is not None and tfidf_matrix is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        vec = tfidf_vectorizer.transform([metin.lower()])
        scores = cosine_similarity(vec, tfidf_matrix).flatten()
        for i in scores.argsort()[::-1][:5]:
            if scores[i] < 0.01:
                break
            row = df.iloc[i]
            benzer_siirler.append({
                "baslik": row.get("baslik", ""),
                "donem": row.get("donem_final", ""),
                "skor": round(float(scores[i]), 3),
                "metin_ozet": _metin_ozet(row.get("metin", ""), 120),
            })

    donem_istatistik = {}
    if df is not None:
        for d in ["Erken Dönem (1947–1962)", "Olgunluk Dönemi (1963–1972)", "Son Dönem (1973–1984)"]:
            alt = df[df["donem_final"] == d]
            if len(alt) > 0:
                donem_istatistik[d] = {
                    "siir_sayisi": int(len(alt)),
                    "ort_kelime": round(float(alt["kelime_sayisi"].mean()), 1),
                }

    return jsonify({
        "baslik": baslik,
        "duygu": skor_data,
        "temalar": tema_data,
        "donem": donem_data,
        "kelime_sayisi": kelime_sayisi,
        "satir_sayisi": satir_sayisi,
        "top_kelimeler": top_kelimeler,
        "top_bigramlar": top_bigramlar,
        "benzer_siirler": benzer_siirler,
        "donem_istatistik": donem_istatistik,
    })


if __name__ == "__main__":
    load_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
