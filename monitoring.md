# Cloud Monitoring Setup — Ticket Tagger

## Overview

Monitoring menggunakan **GCP Cloud Monitoring** (Opsi 1: Console-based, no-code).  
Service: `ticket-tagger` | Region: `asia-southeast1` | Free tier.

**Live URL:** https://ticket-tagger-423883733078.asia-southeast1.run.app/

---

## Checklist Setup

- [ ] Step 1 — Uptime Check
- [ ] Step 2 — Alert Policy: High Latency
- [ ] Step 3 — Alert Policy: Error Rate (5xx)
- [ ] Step 4 — Dashboard (opsional)

---

## Step 1 — Uptime Check

Mendeteksi app down secara otomatis.

1. Buka [console.cloud.google.com](https://console.cloud.google.com)
2. Menu → **Monitoring** → **Uptime checks** → **Create Uptime Check**
3. Isi form:
   ```
   Protocol       : HTTPS
   Resource       : URL
   Hostname       : ticket-tagger-423883733078.asia-southeast1.run.app
   Path           : /
   Check Interval : 5 minutes
   ```
4. Klik **Continue** → bagian **Alert & Notification**, buat alert baru:
   ```
   Name        : ticket-tagger-down
   Notify via  : Email → (email kamu)
   ```
5. Klik **Create**

---

## Step 2 — Alert Policy: High Latency

Notifikasi jika response time > 5 detik.

1. Menu → **Monitoring** → **Alerting** → **Create Policy**
2. **Select a metric** → cari:
   ```
   Cloud Run Revision → Request Latencies
   ```
3. **Filter**: `service_name = ticket-tagger`
4. **Configure alert trigger**:
   ```
   Condition type : Threshold
   Threshold      : 5000 ms (5 detik)
   For            : 2 consecutive violations
   ```
5. **Notification** → tambahkan email
6. **Name**: `ticket-tagger-high-latency`
7. Klik **Create Policy**

---

## Step 3 — Alert Policy: Error Rate (5xx)

Notifikasi jika terjadi lonjakan error server.

1. **Alerting** → **Create Policy**
2. **Select a metric**:
   ```
   Cloud Run Revision → Request Count
   ```
3. **Filter**:
   ```
   service_name        = ticket-tagger
   response_code_class = 5xx
   ```
4. **Configure trigger**:
   ```
   Condition type : Threshold
   Threshold      : 5 requests
   For            : 1 minute
   ```
5. **Name**: `ticket-tagger-error-rate`
6. Klik **Create Policy**

---

## Step 4 — Dashboard (Opsional)

Visualisasi metrics untuk keperluan monitoring & dokumentasi OKR.

1. **Monitoring** → **Dashboards** → **Create Dashboard**
2. **Name**: `Ticket Tagger - Overview`
3. Add widgets:
   - Line chart → `Request Count` (filter: service=ticket-tagger)
   - Line chart → `Request Latencies`
   - Scorecard → `Instance Count`
4. **Save**

---

## Pricing

| Fitur | Free Tier |
|-------|-----------|
| Uptime Checks | Gratis |
| Alerting Policies | Gratis |
| Metrics ingestion | 150 MiB/bulan gratis |
| Dashboards | Gratis |

Untuk project traffic rendah (personal/portofolio), semua langkah di atas berada dalam free tier.

---

## OKR Progress

| Key Result | Status |
|------------|--------|
| KR1 — Python Data Analytics Project | ✅ Done |
| KR2 — Deploy ML Model on GCP | ✅ Done (monitoring pending checklist atas) |
| KR3 — Generative AI Experience | 🔄 In Progress |

### KR2 Detail
| Action Item | Status |
|-------------|--------|
| Define objectives & develop ML model | ✅ Done |
| Evaluate model (F1: 82.2%, target ≥75%) | ✅ Done |
| Prepare for deployment (Dockerfile, cloudbuild.yaml) | ✅ Done |
| Setup GCP Project | ✅ Done |
| Deploy to Cloud Run | ✅ Done — live di asia-southeast1 |
| Test & validate deployment | ✅ Done |
| Implement monitoring tools | 🔄 In Progress (ikuti checklist atas) |

### KR3 Detail
| Action Item | Status |
|-------------|--------|
| Complete hands-on course | ✅ Done — Udemy: "GCP Associate Cloud Engineer - Google Cloud Certification" (bersertifikat) |
| Build & deploy GenAI use case project | 🔄 In Progress — AI Reply Suggester (Gemini 2.5 Flash) sudah berjalan lokal, pending deploy ke Cloud Run |
| Write blog / knowledge sharing session | ❌ Pending |
