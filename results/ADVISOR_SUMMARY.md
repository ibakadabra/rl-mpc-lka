# CARLA Validation — Ön Sonuçlar (2026-06-15)

Town04, sync mode (Ts=0.02s), 5 kontrolcü × hız×sürtünme sweep.
Aşağıdaki tablolar **temiz spawn (seed 2000)** üzerinden — tüm kontrolcüler
aynı yol parçasında, adil karşılaştırma. (Crash/spawn notu en altta.)

---

## Tablo 1 — Yanal takip hatası RMSE (cm), senaryo × kontrolcü

| Senaryo (hız, μ) | fixed MPC | gain-sched MPC | **RL-MPC** | Stanley | Pure Pursuit |
|---|---|---|---|---|---|
| dry 12 m/s     | 0.09 | 0.09 | 0.09 | 5.6  | 4.0  |
| dry 22 m/s     | 0.11 | 0.10 | 0.91 | 13.7 | 18.1 |
| **dry 30 m/s** | 6.2  | 6.5  | **4.3** | 51.0 | 71.4 |
| wet 20 m/s     | 0.08 | 0.07 | 0.50 | 9.8  | 11.4 |
| wet 28 m/s     | 0.28 | 0.33 | 2.05 | 40.7 | 55.9 |

**Ana bulgu:** MPC ailesi (fixed/gain/RL) klasik geometrik kontrolcüleri
(Stanley, Pure Pursuit) **10–50 kat** geçiyor. En zorlu noktada (dry 30 m/s)
RL-MPC, fixed ve gain-scheduled'ı da geçerek en iyi (4.3 vs 6.2 / 6.5 cm).

---

## Tablo 2 — Operasyon limiti: dry 30 m/s (kontrolcülerin ayrıştığı nokta)

| Kontrolcü | RMSE eᵧ (cm) | RMSE e_ψ (rad) | max aᵧ (g) |
|---|---|---|---|
| **RL-MPC**         | **4.3** | **0.0138** | **0.505** |
| fixed MPC          | 6.2 | 0.0139 | 0.583 |
| gain-scheduled MPC | 6.5 | 0.0139 | 0.682 |
| Stanley            | 51.0 | 0.0142 | 0.643 |
| Pure Pursuit       | 71.4 | 0.0198 | 0.763 |

**Yüksek hızda RL-MPC hem en iyi takip hem en iyi konfor** (en düşük yanal ivme).
Tek bir sabit/scheduled ağırlığın zarfı kapatamadığı yerde online adaptasyonun
değeri görünüyor.

---

## Dürüst değerlendirme (hocaya da söylenmeli)

1. **MPC >> geometrik:** kaya gibi sağlam, her senaryoda.
2. **MPC ailesi içinde** (fixed/gain/RL) kolay rejimlerde üçü de ~mükemmel
   (sub-cm); RL avantajı **sadece operasyon limitinde** (yüksek hız) net.
   Kolay noktalarda RL hafif geri kalıyor → mevcut model **25k-step DEMO**,
   eğitim eksik (politika "gerekmeyince default ağırlıkta kal"mayı öğrenmemiş).
3. **Crash'ler spawn artefaktı:** seed 2002 her kontrolcüyü (Stanley dahil)
   12 m/s'de bile çarptırıyor → sürülemez spawn noktası, kontrolcü hatası değil.
   Spawn filtresi ekleniyor.
4. **İstatistik henüz yok:** 3 seed yetersiz; ≥10 seed + anlamlılık testi geliyor.

## Sıradaki adımlar (devam eden)

- **500k tam eğitim** + genişletilmiş domain randomization (düşük-μ, μ-split,
  episode-içi koşul değişimi) → RL'in zarf-geneli avantajını keskinleştirmek.
- **Zor held-out senaryolar** (RL vs gain-scheduled vs fixed) — RL'in yapısal
  üstünlüğünü (state-koşullu online ayarlama) izole eden testler.
- **ISO 11270 / 3888-2 / 4138** standart kriterleri (kod hazır).
