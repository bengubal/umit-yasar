"""
Ümit Yaşar Oğuzcan - Görselleştirme
=====================================
analiz.py çıktısını (umit_yasar_analiz.csv) kullanır.
Üretilen görseller: umit_yasar_gorseller.png

Çalıştır: python gorseller.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

DONEM_RENK = {
    "Erken Dönem (1947–1962)":        "#4C9BE8",
    "Olgunluk Dönemi (1963–1972)":    "#F0A500",
    "Son Dönem (1973–1984)":          "#E05C5C",
    "Belirsiz":                        "#AAAAAA",
}
DONEM_SIRA = [
    "Erken Dönem (1947–1962)",
    "Olgunluk Dönemi (1963–1972)",
    "Son Dönem (1973–1984)",
    "Belirsiz",
]


def yukle():
    for yol in ["umit_yasar_analiz.csv", "/mnt/user-data/outputs/umit_yasar_analiz.csv"]:
        try:
            df = pd.read_csv(yol)
            # donem_final varsa onu kullan (dönem eşleştirmeli CSV)
            if 'donem_final' in df.columns:
                df['donem'] = df['donem_final']
            # Belirsiz dönem kalmamışsa DONEM_SIRA'dan çıkar
            global DONEM_SIRA
            DONEM_SIRA = [d for d in DONEM_SIRA if d != "Belirsiz" or d in df['donem'].values]
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError("umit_yasar_analiz.csv bulunamadı. Önce analiz.py çalıştırın.")


def grafik_1_donem_duygu_dag(ax, df):
    """Dönem × Duygu etiketi yığılmış bar."""
    donemler = [d for d in DONEM_SIRA if d in df['donem'].values]
    etiketler = ["pozitif", "nötr", "negatif"]
    renkler   = ["#4CAF50", "#FFC107", "#F44336"]

    veriler = {}
    for d in donemler:
        alt = df[df['donem'] == d]['st_etiket'].value_counts(normalize=True)
        veriler[d] = {e: alt.get(e, 0) for e in etiketler}

    x = np.arange(len(donemler))
    w = 0.5
    alt = np.zeros(len(donemler))
    for etiket, renk in zip(etiketler, renkler):
        degerler = [veriler[d][etiket] for d in donemler]
        ax.bar(x, degerler, w, bottom=alt, color=renk, label=etiket, alpha=0.85)
        alt += np.array(degerler)

    ax.set_xticks(x)
    ax.set_xticklabels([d.split('(')[0].strip() for d in donemler], fontsize=8)
    ax.set_ylabel("Oran")
    ax.set_title("Dönem × Duygu Dağılımı", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)


def grafik_2_ort_skor(ax, df):
    """Dönem bazlı ortalama SentiTurkNet skoru + hata çubuğu."""
    donemler = [d for d in DONEM_SIRA if d in df['donem'].values]
    skorlar  = [df[df['donem'] == d]['st_skor'].mean() for d in donemler]
    hatalar  = [df[df['donem'] == d]['st_skor'].std() for d in donemler]
    renkler  = [DONEM_RENK[d] for d in donemler]

    x = np.arange(len(donemler))
    bars = ax.bar(x, skorlar, 0.5, yerr=hatalar, color=renkler, alpha=0.85,
                  error_kw={"ecolor": "gray", "capsize": 4})
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([d.split('(')[0].strip() for d in donemler], fontsize=8)
    ax.set_ylabel("Ortalama Duygu Skoru")
    ax.set_title("Dönem Bazlı Duygu Skoru", fontweight='bold')

    for bar, skor in zip(bars, skorlar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{skor:.3f}", ha='center', va='bottom', fontsize=8)


def grafik_3_tema_dag(ax, df):
    """Tüm şiirlerde dominant tema dağılımı pasta grafik."""
    tema_map = {
        "ask_ozlem": "Aşk & Özlem",
        "olum_aci":  "Ölüm & Acı",
        "yalnizlik": "Yalnızlık",
        "umut":      "Umut",
        "doga":      "Doğa",
        "toplum":    "Toplum",
        "genel":     "Genel",
    }
    sayilar = df['dominant_tema'].value_counts()
    etiketler = [tema_map.get(t, t) for t in sayilar.index]
    renkler = ["#E91E63","#9C27B0","#3F51B5","#009688","#8BC34A","#FF5722","#9E9E9E"]

    wedges, texts, autotexts = ax.pie(
        sayilar.values,
        labels=etiketler,
        autopct='%1.0f%%',
        colors=renkler[:len(sayilar)],
        startangle=140,
        textprops={'fontsize': 8},
    )
    ax.set_title("Tema Dağılımı (300 Şiir)", fontweight='bold')


def grafik_4_donem_tema(ax, df):
    """Dönem × Tema ısı haritası."""
    donemler = [d for d in DONEM_SIRA if d in df['donem'].values and d != "Belirsiz"]
    tema_cols = [c for c in df.columns if c.startswith('tema_')]
    tema_isimleri = {
        "tema_ask_ozlem": "Aşk & Özlem",
        "tema_olum_aci":  "Ölüm & Acı",
        "tema_yalnizlik": "Yalnızlık",
        "tema_umut":      "Umut",
        "tema_doga":      "Doğa",
        "tema_toplum":    "Toplum",
    }

    matris = []
    for d in donemler:
        alt = df[df['donem'] == d]
        satir = [alt[tc].mean() for tc in tema_cols if tc in tema_isimleri]
        matris.append(satir)

    gecerli_cols = [tc for tc in tema_cols if tc in tema_isimleri]
    matris = np.array(matris)

    im = ax.imshow(matris, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(gecerli_cols)))
    ax.set_xticklabels([tema_isimleri[tc] for tc in gecerli_cols], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(donemler)))
    ax.set_yticklabels([d.split('(')[0].strip() for d in donemler], fontsize=8)
    ax.set_title("Dönem × Tema Yoğunluğu", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)


def grafik_5_kelime_dag(ax, df):
    """Dönem bazlı kelime sayısı kutu grafiği."""
    donemler = [d for d in DONEM_SIRA if d in df['donem'].values]
    veriler  = [df[df['donem'] == d]['kelime_sayisi'].values for d in donemler]
    renkler  = [DONEM_RENK[d] for d in donemler]

    bp = ax.boxplot(veriler, patch_artist=True, notch=False)
    for patch, renk in zip(bp['boxes'], renkler):
        patch.set_facecolor(renk)
        patch.set_alpha(0.7)

    ax.set_xticklabels([d.split('(')[0].strip() for d in donemler], fontsize=8)
    ax.set_ylabel("Kelime Sayısı")
    ax.set_title("Dönem Bazlı Şiir Uzunluğu", fontweight='bold')


def grafik_6_poz_neg_trend(ax, df):
    """Bilinen dönemlerde pozitif/negatif oran çizgi grafiği."""
    donemler = [d for d in DONEM_SIRA if d in df['donem'].values and d != "Belirsiz"]
    x_labels = [d.split('(')[0].strip() for d in donemler]
    x = range(len(donemler))

    poz = [df[df['donem'] == d]['st_etiket'].eq('pozitif').mean() for d in donemler]
    neg = [df[df['donem'] == d]['st_etiket'].eq('negatif').mean() for d in donemler]
    notr= [df[df['donem'] == d]['st_etiket'].eq('nötr').mean() for d in donemler]

    ax.plot(x, poz,  'o-', color='#4CAF50', label='Pozitif', linewidth=2, markersize=8)
    ax.plot(x, neg,  's-', color='#F44336', label='Negatif', linewidth=2, markersize=8)
    ax.plot(x, notr, '^-', color='#FFC107', label='Nötr',    linewidth=2, markersize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Oran")
    ax.set_ylim(0, 1)
    ax.set_title("Dönemler Arası Duygu Trendi", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    df = yukle()
    print(f"Yüklendi: {len(df)} şiir")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Ümit Yaşar Oğuzcan Şiirlerinde Duygusal Evrim Analizi\n"
        "(SentiTurkNet + Tema Tespiti)",
        fontsize=14, fontweight='bold', y=1.01
    )

    grafik_1_donem_duygu_dag(axes[0, 0], df)
    grafik_2_ort_skor(axes[0, 1], df)
    grafik_3_tema_dag(axes[0, 2], df)
    grafik_4_donem_tema(axes[1, 0], df)
    grafik_5_kelime_dag(axes[1, 1], df)
    grafik_6_poz_neg_trend(axes[1, 2], df)

    plt.tight_layout()
    out = "/mnt/user-data/outputs/umit_yasar_gorseller.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✓ Görseller kaydedildi: {out}")
    plt.close()


if __name__ == "__main__":
    main()
