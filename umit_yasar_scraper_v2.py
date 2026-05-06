"""
Ümit Yaşar Oğuzcan Şiir Scraper v2 - antoloji.com
===================================================
Düzeltmeler:
  - Doğru selector: div.pd-text (metin), div.pd-title (başlık)
  - Duplikasyon temizleme (her sayfada tekrar eden 5 reklam şiiri)
  - Yabancı şair filtresi
  - Şiir adından kitap eşleştirme
Çalıştır: python umit_yasar_scraper_v2.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Kitap → Yıl ──────────────────────────────────────────────────────────────
KITAP_YILLARI = {
    "İnsanoğlu": 1947,
    "Deniz Musikisi": 1949,
    "Dillere Destan": 1954,
    "Dolmuş": 1955,
    "Aşkımızın Son Çarşambası": 1955,
    "Bir Daha Ölmek": 1956,
    "Kör Ayna": 1957,
    "İki Kişiye Bir Dünya": 1957,
    "Beni Unutma": 1959,
    "Karanlığın Gözleri": 1960,
    "Akıllı Maymunlar": 1960,
    "Seninle Ölmek İstiyorum": 1960,
    "Üstüme Varma İstanbul": 1961,
    "Sahibini Arayan Mektuplar": 1961,
    "Yeni Dünya Rekoru": 1961,
    "Sevenler Ölmez": 1962,
    "Çigan Gözler": 1962,
    "Ötesi Yok": 1963,
    "Hüzün Şarkıları": 1963,
    "Bir Gün Anlarsın": 1965,
    "Sadrazamın Sol Kulağı": 1965,
    "Mihriban'a Şiirler": 1965,
    "Taşlar ve Başlar": 1966,
    "Seni Sevmek": 1966,
    "Toprak Olana Kadar": 1968,
    "Ben Seni Sevdim mi": 1968,
    "Halktan Yana": 1969,
    "Aşk mıydı O": 1969,
    "Önce Sen Sonra Sen": 1971,
    "Rubailer": 1972,
    "Yalan Bitti": 1975,
    "Acılar Denizi": 1977,
    "En Eski Yalnızlığımdır Aşk Benim": 1978,
    "Dikiz Aynası": 1982,
    "Bütün Şiirleri": 1982,
}

# Şiir adı → kitap eşleştirme tablosu (bilinen şiirler)
SIIR_KITAP = {
    "Bir Gün Anlarsın": "Bir Gün Anlarsın",
    "Beni Unutma": "Beni Unutma",
    "Acılar Denizi": "Acılar Denizi",
    "Seninle Ölmek İstiyorum": "Seninle Ölmek İstiyorum",
    "Dost Bildiklerim": "Bir Gün Anlarsın",
    "Çigan Gözler": "Çigan Gözler",
    "Karanlığın Gözleri": "Karanlığın Gözleri",
    "Galata Kulesi": "Acılar Denizi",
    "Oğluma Ağıt": "Acılar Denizi",
    "Oğul Koşması": "Acılar Denizi",
    "Deniz Musikisi": "Deniz Musikisi",
    "Üstüme Varma İstanbul": "Üstüme Varma İstanbul",
    "Ötesi Yok": "Ötesi Yok",
    "Yalan Bitti": "Yalan Bitti",
    "Rubailer": "Rubailer",
    "Dörtlükler": "Rubailer",
    "Ben Seni Sevdim mi": "Ben Seni Sevdim mi",
    "Aşkmıydı O": "Aşk mıydı O",
    "İki Kişiye Bir Dünya": "İki Kişiye Bir Dünya",
    "En Eski Yalnızlığımdır Aşk Benim": "En Eski Yalnızlığımdır Aşk Benim",
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-zçğıöşü0-9\s]", "", s.lower()).strip()

def kitap_yil_bul(siir_baslik: str) -> tuple:
    """Şiir başlığından kitap ve yıl bul."""
    b = _norm(siir_baslik)

    # 1. Şiir adı → kitap tablosundan
    for siir, kitap in SIIR_KITAP.items():
        if _norm(siir) in b or b in _norm(siir):
            return kitap, KITAP_YILLARI.get(kitap)

    # 2. Şiir adı doğrudan kitap adıyla eşleşiyor mu?
    for kitap, yil in KITAP_YILLARI.items():
        if _norm(kitap) in b or b in _norm(kitap):
            return kitap, yil

    return None, None

def donem_belirle(yil) -> str:
    if yil is None:
        return "Belirsiz"
    if yil <= 1962:
        return "Erken Dönem (1947–1962)"
    if yil <= 1972:
        return "Olgunluk Dönemi (1963–1972)"
    return "Son Dönem (1973–1984)"

# ── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
})
BASE = "https://www.antoloji.com"
LIST_BASE = f"{BASE}/umit-yasar-oguzcan/siirleri/ara-/sirala-/"
DELAY = 2.0

# Tekrar eden reklam şiirleri (her sayfada görünüyor, filtrele)
REKLAM_URL = {
    "https://www.antoloji.com/turk-insanligi-siiri/",
    "https://www.antoloji.com/yasam-insana-ektigini-bictirir-siiri/",
    "https://www.antoloji.com/kutlu-olsun-79-siiri/",
    "https://www.antoloji.com/sensiz-sehir-7-siiri/",
    "https://www.antoloji.com/davet-300-siiri/",
}

# Ümit Yaşar olmayan şairlerin URL pattern'i
YABANCI_SAIR = ["onder-karacay", "ahmet-arslan", "kudret-kobal"]


def get_soup(url: str, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"[{attempt}/{retries}] {url} → {e}")
            if attempt < retries:
                time.sleep(DELAY * attempt)
    log.error(f"Atlandı: {url}")
    return None


def toplam_sayfa() -> int:
    soup = get_soup(LIST_BASE)
    if not soup:
        return 10
    nums = []
    for a in soup.find_all("a", href=re.compile(r"sayfa-\d+")):
        m = re.search(r"sayfa-(\d+)", a["href"])
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 10


def siir_linkleri_cek(sayfa_no: int) -> list:
    url = LIST_BASE if sayfa_no == 1 else f"{LIST_BASE}sayfa-{sayfa_no}/"
    soup = get_soup(url)
    if not soup:
        return []

    siirler = []
    goruldu = set()

    for a in soup.find_all("a", href=re.compile(r"/.*-siiri/")):
        href = a["href"]
        if not href.startswith("http"):
            href = BASE + href

        # Filtreler
        if href in REKLAM_URL:
            continue
        if any(x in href for x in YABANCI_SAIR):
            continue
        if href in goruldu:
            continue

        goruldu.add(href)
        siirler.append({
            "baslik": a.get_text(strip=True),
            "url": href,
        })

    return siirler


def siir_detay_cek(url: str) -> dict:
    soup = get_soup(url)
    if not soup:
        return {}

    # ── Metin: div.pd-text ──
    metin = ""
    el = soup.select_one("div.pd-text")
    if el:
        for br in el.find_all("br"):
            br.replace_with("\n")
        metin = el.get_text(separator="\n").strip()
        # Fazla boş satırları temizle
        metin = re.sub(r"\n{3,}", "\n\n", metin)

    # ── Başlık: div.pd-title ──
    baslik = ""
    el = soup.select_one("div.pd-title")
    if el:
        baslik = el.get_text(strip=True)
        # "Bir Gün Anlarsın Şiiri - Ümit Yaşar Oğuzcan" → "Bir Gün Anlarsın"
        baslik = re.sub(r"\s+[Şş]iiri.*$", "", baslik).strip()
        baslik = re.sub(r"\s*-\s*Ümit Yaşar.*$", "", baslik).strip()

    # ── Şairi doğrula ──
    sair_el = soup.select_one("div.pb-title a, div.poet-bar a")
    sair = sair_el.get_text(strip=True) if sair_el else ""

    return {
        "baslik_sayfa": baslik,
        "metin": metin,
        "sair": sair,
    }


def main():
    output_path = Path("umit_yasar_siirler_v2.csv")

    log.info("Sayfa sayısı kontrol ediliyor...")
    n_sayfa = toplam_sayfa()
    log.info(f"Toplam {n_sayfa} sayfa.")

    # 1. Linkleri topla (URL bazlı tekilleştirme)
    tum_linkler = {}
    for sayfa in range(1, n_sayfa + 1):
        log.info(f"Liste sayfası {sayfa}/{n_sayfa}...")
        for s in siir_linkleri_cek(sayfa):
            if s["url"] not in tum_linkler:
                tum_linkler[s["url"]] = s
        time.sleep(DELAY)

    siir_listesi = list(tum_linkler.values())
    log.info(f"\nTekil şiir sayısı: {len(siir_listesi)}\n")

    # 2. Detayları çek
    kayitlar = []
    for i, siir in enumerate(siir_listesi, 1):
        log.info(f"[{i}/{len(siir_listesi)}] {siir['baslik'][:50]}")
        detay = siir_detay_cek(siir["url"])

        baslik = detay.get("baslik_sayfa") or siir["baslik"]
        kitap, yil = kitap_yil_bul(baslik)
        donem = donem_belirle(yil)

        kayit = {
            "id": i,
            "baslik": baslik,
            "url": siir["url"],
            "metin": detay.get("metin", ""),
            "sair": detay.get("sair", "Ümit Yaşar Oğuzcan"),
            "kitap_adi": kitap or "",
            "yayin_yili": yil,
            "donem": donem,
            "kelime_sayisi": len(detay.get("metin", "").split()),
        }
        kayitlar.append(kayit)

        if i % 25 == 0:
            pd.DataFrame(kayitlar).to_csv(output_path, index=False, encoding="utf-8-sig")
            log.info(f"  ✓ Ara kayıt ({i} şiir)")

        time.sleep(DELAY)

    # 3. Kaydet
    df = pd.DataFrame(kayitlar)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 4. Özet
    print("\n" + "="*60)
    print(f"✓ {len(df)} şiir → {output_path}")
    print(f"\nMetni dolu olan : {(df['kelime_sayisi'] > 0).sum()}")
    print(f"Metni boş olan  : {(df['kelime_sayisi'] == 0).sum()}")
    print(f"\nDönem dağılımı:\n{df['donem'].value_counts().to_string()}")
    print(f"\nKitap bilgisi var: {df['yayin_yili'].notna().sum()}")
    print(f"Kitap bilgisi yok: {df['yayin_yili'].isna().sum()}")
    print(f"\nOrtalama kelime : {df['kelime_sayisi'].mean():.1f}")
    print("="*60)

    # Örnek 3 şiir göster
    print("\n--- ÖRNEK ŞİİR (ilk metin dolu) ---")
    ornek = df[df["kelime_sayisi"] > 0].iloc[0]
    print(f"Başlık : {ornek['baslik']}")
    print(f"Dönem  : {ornek['donem']}")
    print(f"Metin  :\n{ornek['metin'][:300]}")

    return df


if __name__ == "__main__":
    main()
