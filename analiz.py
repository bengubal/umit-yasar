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


def sentiturk_skor(metin: str) -> dict:
    """
    Metin için SentiTurkNet benzeri sözcük sayım skoru hesapla.
    Returns: {pozitif, negatif, notör, skor, etiket}
    """
    if not isinstance(metin, str) or not metin.strip():
        return {"pozitif": 0, "negatif": 0, "notr": 0, "skor": 0.0, "etiket": "nötr"}

    # Tokenize: küçük harf, noktalama temizle
    kelimeler = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", metin.lower()).split()
    
    poz = 0.0
    neg = 0.0
    
    for k in kelimeler:
        if k in GUCLU_POZITIF:
            poz += 2.0
        elif k in POZITIF_KELIMELER:
            poz += 1.0
        if k in GUCLU_NEGATIF:
            neg += 2.0
        elif k in NEGATIF_KELIMELER:
            neg += 1.0

    toplam = len(kelimeler) if kelimeler else 1
    skor = (poz - neg) / toplam  # normalize

    if skor > 0.02:
        etiket = "pozitif"
    elif skor < -0.02:
        etiket = "negatif"
    else:
        etiket = "nötr"

    return {
        "st_pozitif_sayi": poz,
        "st_negatif_sayi": neg,
        "st_skor": round(skor, 4),
        "st_etiket": etiket,
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


def berturk_duygu(metinler: list, batch_size: int = 8) -> list:
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
            max_length=512,
        )
        print("  Model yüklendi ✓")

        sonuclar = []
        for i in range(0, len(metinler), batch_size):
            batch = metinler[i:i+batch_size]
            # Çok uzun metinleri kısalt
            batch = [m[:1000] if isinstance(m, str) else "" for m in batch]
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

            if (i // batch_size + 1) % 5 == 0:
                print(f"  BERTurk: {min(i+batch_size, len(metinler))}/{len(metinler)} şiir işlendi")

        return sonuclar

    except Exception as e:
        print(f"  BERTurk yüklenemedi: {e}")
        print("  SentiTurkNet sonuçları kullanılacak.")
        return [{"bert_etiket": "yok", "bert_skor": 0.0}] * len(metinler)


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

    # ── 3. BERTurk ───────────────────────────────────────────────────────────
    print("\n[3/3] BERTurk duygu analizi yapılıyor...")
    bert_sonuclar = berturk_duygu(df['metin'].tolist())
    bert_df = pd.DataFrame(bert_sonuclar)
    df = pd.concat([df, bert_df], axis=1)
    if 'bert_etiket' in df.columns and df['bert_etiket'].iloc[0] != 'yok':
        print(f"  BERTurk dağılım: {df['bert_etiket'].value_counts().to_dict()}")

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
    out = "/mnt/user-data/outputs/umit_yasar_analiz.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✓ Analiz CSV kaydedildi: {out}")

    ozet_out = "/mnt/user-data/outputs/umit_yasar_donem_ozet.csv"
    donem_ozet.to_csv(ozet_out, encoding="utf-8-sig")
    print(f"✓ Dönem özeti kaydedildi: {ozet_out}")

    return df, donem_ozet


if __name__ == "__main__":
    df, ozet = main()
