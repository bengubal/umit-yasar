"""
Ümit Yaşar Oğuzcan Şiir Scraper - antoloji.com
================================================
Tüm şiirleri çeker, kitap bilgisi ve döneme göre etiketler,
CSV olarak kaydeder.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging
from pathlib import Path

# ── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Kitap → Yıl Sözlüğü ──────────────────────────────────────────────────────
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

# Normalize edilmiş lookup (küçük harf, noktalama yok) → orijinal isim
_KITAP_NORM = {
    re.sub(r"[^a-zçğıöşü0-9]", "", k.lower()): k
    for k in KITAP_YILLARI
}

def _norm(s: str) -> str:
    """Kitap adı normalleştirme (küçük harf + özel karakterleri at)."""
    return re.sub(r"[^a-zçğıöşü0-9]", "", s.lower())


def kitap_yil_bul(kitap_adi: str) -> tuple[str | None, int | None]:
    """
    Kitap adından (kısmi eşleşme dahil) yayın yılını döndür.
    Returns: (normalize_edilmiş_kitap_adı, yıl) veya (None, None)
    """
    if not kitap_adi:
        return None, None
    n = _norm(kitap_adi)
    # Tam eşleşme
    if n in _KITAP_NORM:
        k = _KITAP_NORM[n]
        return k, KITAP_YILLARI[k]
    # Kısmi eşleşme
    for anahtar, orijinal in _KITAP_NORM.items():
        if n in anahtar or anahtar in n:
            return orijinal, KITAP_YILLARI[orijinal]
    return kitap_adi, None


def donem_belirle(yil: int | None) -> str:
    """Yıla göre edebi dönem etiketini döndür."""
    if yil is None:
        return "Belirsiz"
    if yil <= 1962:
        return "Erken Dönem (1947–1962)"
    if yil <= 1972:
        return "Olgunluk Dönemi (1963–1972)"
    return "Son Dönem (1973–1984)"


# ── HTTP Oturumu ──────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; AkademikArastirma/1.0; "
        "text-mining-project/umit-yasar)"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

BASE = "https://www.antoloji.com"
LIST_BASE = f"{BASE}/umit-yasar-oguzcan/siirleri/ara-/sirala-/"
DELAY = 2.0  # saniye — sunucuya saygılı olalım


def get_soup(url: str, retries: int = 3) -> BeautifulSoup | None:
    """URL'yi çek ve BeautifulSoup döndür; hata olursa tekrar dene."""
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


# ── Liste Sayfaları ───────────────────────────────────────────────────────────
def toplam_sayfa_sayisi() -> int:
    """İlk liste sayfasından toplam sayfa sayısını çek."""
    soup = get_soup(LIST_BASE)
    if not soup:
        return 10  # varsayılan
    # Sayfalama linklerindeki en büyük sayıyı bul
    nums = []
    for a in soup.select("ul.pagination a, div.pagination a, a[href*='sayfa-']"):
        m = re.search(r"sayfa-(\d+)", a.get("href", ""))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 10


def siir_linkleri_cek(sayfa_no: int) -> list[dict]:
    """Bir liste sayfasından şiir başlıklarını ve URL'lerini çek."""
    if sayfa_no == 1:
        url = LIST_BASE
    else:
        url = f"{LIST_BASE}sayfa-{sayfa_no}/"
    
    soup = get_soup(url)
    if not soup:
        return []
    
    siirler = []
    # Her li elementi: numara, başlık linki, tarih
    for li in soup.select("ul.poem-list li, ul li"):
        a = li.find("a", href=re.compile(r"/.*-siiri/"))
        if not a:
            continue
        href = a.get("href", "")
        if not href.startswith("http"):
            href = BASE + href
        baslik = a.get_text(strip=True)
        # Siteye eklenme tarihi (projen için değil ama saklayalım)
        tarih_el = li.find("span") or li.find("time")
        tarih = tarih_el.get_text(strip=True) if tarih_el else ""
        siirler.append({
            "baslik": baslik,
            "url": href,
            "site_tarihi": tarih,
        })
    return siirler


# ── Şiir Sayfası ─────────────────────────────────────────────────────────────
def siir_detay_cek(url: str) -> dict:
    """
    Tek şiir sayfasından metin, kitap adı ve diğer meta verileri çek.
    """
    soup = get_soup(url)
    if not soup:
        return {}

    # ── Şiir metni ──
    metin = ""
    for selector in [
        "div.poem-detail-text",
        "div.siir-icerik",
        "div#poem-text",
        "div.poem-text",
        "div[itemprop='description']",
    ]:
        el = soup.select_one(selector)
        if el:
            # <br> → satır sonu
            for br in el.find_all("br"):
                br.replace_with("\n")
            metin = el.get_text(separator="\n").strip()
            break

    # ── Kitap adı ──
    kitap_adi = ""
    for selector in [
        "a[href*='/kitap/']",
        "span.book-name",
        "div.poem-book a",
        "div.siir-kitap a",
        "li:contains('Kitap')",
    ]:
        try:
            el = soup.select_one(selector)
        except Exception:
            continue
        if el:
            kitap_adi = el.get_text(strip=True)
            break
    
    # Kitap bilgisi bulunamadıysa meta/breadcrumb'dan dene
    if not kitap_adi:
        for el in soup.find_all(string=re.compile(r"Kitap\s*:", re.I)):
            parent = el.parent
            if parent:
                text = parent.get_text()
                m = re.search(r"Kitap\s*:\s*(.+)", text, re.I)
                if m:
                    kitap_adi = m.group(1).strip()
                    break

    # ── Şair adı (kontrol) ──
    sair = ""
    el = soup.select_one("a[href*='/umit-yasar-oguzcan/']")
    if el:
        sair = el.get_text(strip=True)

    # ── Şiir başlığı ──
    baslik = ""
    for sel in ["h1.poem-title", "h1", "h2.poem-title"]:
        el = soup.select_one(sel)
        if el:
            baslik = el.get_text(strip=True)
            break

    return {
        "baslik_sayfa": baslik,
        "metin": metin,
        "kitap_adi_ham": kitap_adi,
        "sair": sair,
    }


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────
def main():
    output_path = Path("/mnt/user-data/outputs/umit_yasar_siirler.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Toplam sayfa sayısı kontrol ediliyor...")
    n_sayfa = toplam_sayfa_sayisi()
    log.info(f"Toplam {n_sayfa} sayfa bulundu.")

    # 1. Tüm şiir linklerini topla
    tum_siirler = []
    for sayfa in range(1, n_sayfa + 1):
        log.info(f"Liste sayfası {sayfa}/{n_sayfa} işleniyor...")
        siirler = siir_linkleri_cek(sayfa)
        tum_siirler.extend(siirler)
        log.info(f"  → {len(siirler)} şiir bulundu (toplam: {len(tum_siirler)})")
        time.sleep(DELAY)

    log.info(f"\nToplam {len(tum_siirler)} şiir linki toplandı. Detaylar çekiliyor...\n")

    # 2. Her şiirin detayını çek
    kayitlar = []
    for i, siir in enumerate(tum_siirler, 1):
        log.info(f"[{i}/{len(tum_siirler)}] {siir['baslik'][:50]}")
        detay = siir_detay_cek(siir["url"])
        
        # Kitap eşleştirme
        kitap_ham = detay.get("kitap_adi_ham", "")
        kitap_norm, yil = kitap_yil_bul(kitap_ham)
        donem = donem_belirle(yil)

        kayit = {
            "id": i,
            "baslik": detay.get("baslik_sayfa") or siir["baslik"],
            "url": siir["url"],
            "metin": detay.get("metin", ""),
            "kitap_adi": kitap_norm or kitap_ham or "",
            "yayin_yili": yil,
            "donem": donem,
            "metin_uzunluk": len(detay.get("metin", "")),
            "kelime_sayisi": len(detay.get("metin", "").split()),
        }
        kayitlar.append(kayit)

        # Her 50 şiirde bir ara kayıt yap
        if i % 50 == 0:
            pd.DataFrame(kayitlar).to_csv(output_path, index=False, encoding="utf-8-sig")
            log.info(f"  ✓ Ara kayıt: {i} şiir kaydedildi.")

        time.sleep(DELAY)

    # 3. Final CSV
    df = pd.DataFrame(kayitlar)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 4. Özet
    log.info("\n" + "="*60)
    log.info(f"✓ Toplam {len(df)} şiir kaydedildi → {output_path}")
    log.info(f"\nDönem dağılımı:")
    log.info(df["donem"].value_counts().to_string())
    log.info(f"\nKitap bilgisi bulunan: {df['yayin_yili'].notna().sum()} şiir")
    log.info(f"Kitap bilgisi eksik:   {df['yayin_yili'].isna().sum()} şiir")
    log.info(f"\nOrtalama kelime sayısı: {df['kelime_sayisi'].mean():.1f}")
    log.info("="*60)

    return df


if __name__ == "__main__":
    main()
