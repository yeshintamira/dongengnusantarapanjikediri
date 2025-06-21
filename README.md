# 🏰 Aplikasi Dongeng Nusantara Panji Kediri

Aplikasi ini merupakan sistem pencarian dongeng Bahasa Indonesia berbasis topik menggunakan metode *Latent Dirichlet Allocation (LDA)* dan *Latent Semantic Indexing (LSI)*. Pengguna dapat mencari dongeng berdasarkan kata kunci atau topik, dan sistem akan merekomendasikan dongeng yang paling relevan berdasarkan model yang dipilih.

## 📖 Cara Menggunakan Aplikasi

1. **Buka Aplikasi**
   - Akses aplikasi melalui URL berikut:  
     👉 https://dongeng-nusantara-panji-kediri.streamlit.app/

2. **Masukkan Kata Kunci atau Topik**
   - Ketikkan topik atau kata kunci yang ingin dicari, seperti:
     - `cinta kasih sayang`
     - `putri kerajaan`
     - `persahabatan`

3. **Pilih Metode Pencarian**
   - Klik salah satu dari dua tombol pencarian yang tersedia:
     - 🔍 **Cari dengan LDA** — menggunakan model **Latent Dirichlet Allocation**, cocok untuk rekomendasi berbasis distribusi topik.
     - 🔍 **Cari dengan LSI** — menggunakan model **Latent Semantic Indexing**, cocok untuk pencarian berbasis kemiripan makna.

4. **Lihat Hasil Pencarian**
   - Sistem akan menampilkan **5 dongeng paling relevan** berdasarkan metode yang dipilih.
   - Ditampilkan bersama **judul** dan **isi cerita**.

## 🧠 Tentang Dua Metode Pencarian

Aplikasi menyediakan dua metode pencarian:

- **LDA (Latent Dirichlet Allocation)**  
  Model probabilistik yang mengidentifikasi distribusi topik dalam dokumen. Cocok untuk mengenali struktur tematik dongeng.

- **LSI (Latent Semantic Indexing)**  
  Model berbasis vektor semantik yang mengungkap hubungan tersembunyi antar kata dan dokumen. Cocok untuk pencarian berbasis kemiripan makna.

Pengguna bebas memilih metode sesuai kebutuhan eksplorasi dongeng.
