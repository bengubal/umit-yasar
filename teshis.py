"""
Şiir sayfasının gerçek HTML yapısını teşhis eder.
Çalıştır: python teshis.py
Çıktıyı Claude'a yapıştır.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
})

# ── 1. CSV'den ilk birkaç URL'yi oku ──────────────────────────────────────────
try:
    df = pd.read_csv("umit_yasar_siirler.csv")
    urls = df["url"].dropna().head(3).tolist()
    print(f"CSV'den {len(urls)} URL alındı.")
except Exception as e:
    print(f"CSV okunamadı: {e}")
    urls = [
        "https://www.antoloji.com/beni-unutma-siiri/",
        "https://www.antoloji.com/bir-gun-anlarsın-siiri/",
    ]

# ── 2. İlk URL'yi çek ve yapıyı analiz et ─────────────────────────────────────
url = urls[0]
print(f"\nTest URL: {url}\n")
r = SESSION.get(url, timeout=15)
print(f"Status: {r.status_code}")
soup = BeautifulSoup(r.text, "html.parser")

# Tüm div class'larını listele
print("\n=== DIV CLASS'LARI ===")
divs = {}
for d in soup.find_all("div"):
    cls = " ".join(d.get("class", []))
    if cls:
        divs[cls] = divs.get(cls, 0) + 1
for cls, cnt in sorted(divs.items()):
    print(f"  {cnt}x  .{cls}")

# p etiketleri
print("\n=== P ETİKETLERİ (class'lı) ===")
for p in soup.find_all("p"):
    cls = " ".join(p.get("class", []))
    if cls:
        print(f"  .{cls}: {p.get_text(strip=True)[:60]}")

# Tüm <a> href'leri (kitap, book, eser içerenler)
print("\n=== KİTAP LİNKLERİ ===")
for a in soup.find_all("a", href=True):
    href = a["href"]
    if any(x in href.lower() for x in ["kitap", "book", "eser"]):
        print(f"  {href} → {a.get_text(strip=True)[:50]}")

# İlk 2000 karakter ham HTML
print("\n=== HAM HTML (ilk 2000 karakter) ===")
print(r.text[:2000])
