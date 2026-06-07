# Panduan Lengkap: Transformer Architecture

> Dari teori hingga implementasi SLM mini dari scratch dengan PyTorch

---

## Daftar Isi

1. [Teori & Komponen](#1-teori--komponen)
   - [Self-Attention](#11-self-attention)
   - [Multi-Head Attention](#12-multi-head-attention)
   - [Positional Encoding](#13-positional-encoding)
     - [Sinusoidal](#sinusoidal-positional-encoding-vaswani-et-al-2017)
     - [Learned / Trainable](#learned--trainable-positional-encoding)
     - [RoPE](#rope-rotary-position-embedding-su-et-al-2021)
   - [Feed-Forward Network](#14-feed-forward-network)
   - [Layer Normalization & Residual](#15-layer-normalization--residual-connection)
2. [Implementasi PyTorch From Scratch](#2-implementasi-pytorch-from-scratch)
   - [Shared Components & Ketiga PE](#setup-awal-shared-components)
   - [Encoder Only](#21-encoder-only-bert-style)
   - [Decoder Only](#22-decoder-only-gpt-style)
   - [Encoder-Decoder](#23-encoder-decoder-t5-style)
3. [SLM Mini: Model Tanya Jawab Sederhana](#3-slm-mini-model-tanya-jawab-sederhana)

---

## 1. Teori & Komponen

### 1.1 Self-Attention

Self-attention memungkinkan setiap token dalam sebuah sequence untuk "memperhatikan" token lain dalam sequence yang sama, sehingga model bisa memahami konteks dan hubungan antar kata.

#### Konsep Q, K, V

Setiap token direpresentasikan sebagai tiga vektor:
- **Query (Q)** — "Apa yang saya cari?"
- **Key (K)** — "Apa yang saya tawarkan?"
- **Value (V)** — "Informasi apa yang saya bawa?"

Ketiganya dihitung dari embedding input `X` menggunakan proyeksi linear:

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

dimana `W_Q, W_K, W_V ∈ ℝ^(d_model × d_k)` adalah parameter yang dipelajari.

#### Rumus Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V
```

**Penjelasan tiap bagian:**

| Komponen | Penjelasan |
|---|---|
| `Q · K^T` | Dot product → skor kesamaan antar semua pasangan token |
| `/ √d_k` | Scaling agar gradien tidak vanish saat dimensi besar |
| `softmax(...)` | Normalisasi skor menjadi distribusi probabilitas (0–1, jumlah = 1) |
| `· V` | Weighted sum dari value berdasarkan skor attention |

**Contoh skalar kecil:**

Misal kita punya 3 token, `d_k = 4`:
```
Q·K^T = [[9, 2, 1],   → setelah scaling & softmax → [[0.95, 0.03, 0.02],
          [1, 8, 2],                                   [0.03, 0.90, 0.07],
          [1, 2, 7]]                                   [0.04, 0.14, 0.82]]
```
Token pertama sangat memperhatikan dirinya sendiri (0.95), sedikit ke token lain.

#### Causal Mask (untuk Decoder)

Pada decoder, token tidak boleh "melihat" token masa depan. Ini dicapai dengan menambahkan mask `-∞` pada posisi upper-triangle:

```
masked_score[i][j] = score[i][j]  jika j ≤ i
                   = -∞            jika j > i
```

Setelah softmax, nilai `-∞` menjadi `0` sehingga token masa depan diabaikan.

---

### 1.2 Multi-Head Attention

Satu kepala attention hanya bisa belajar satu jenis relasi. Multi-head attention menjalankan beberapa attention secara paralel, lalu menggabungkan hasilnya.

#### Rumus

Untuk `h` buah head:

```
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

dimana:
- `W_Q_i, W_K_i, W_V_i ∈ ℝ^(d_model × d_k)` dengan `d_k = d_model / h`
- `W_O ∈ ℝ^(h·d_v × d_model)` adalah proyeksi output

**Intuisi:** Setiap head belajar aspek berbeda dari relasi antar token. Misalnya:
- Head 1 → relasi sintaktik (subjek-objek)
- Head 2 → relasi semantik (sinonim, antonim)
- Head 3 → relasi posisional (token berdekatan)

#### Dimensi

```
Input: (batch, seq_len, d_model)
Q, K, V per head: (batch, seq_len, d_k) dimana d_k = d_model / n_heads
Output: (batch, seq_len, d_model)
```

---

### 1.3 Positional Encoding

Transformer tidak memiliki urutan bawaan (tidak seperti RNN). Positional encoding menyuntikkan informasi posisi ke dalam embedding. Ada tiga pendekatan utama:

| Metode | Di mana ditambahkan | Dilatih? | Dipakai oleh |
|---|---|---|---|
| Sinusoidal | Ke embedding (`x + PE`) | ❌ Fixed | Original Transformer |
| Learned / Trainable | Ke embedding (`x + PE`) | ✅ Ya | BERT, GPT-2 |
| RoPE | Ke Q & K di dalam attention | ❌ Fixed | LLaMA, Mistral, Gemma |

---

#### Sinusoidal Positional Encoding (Vaswani et al., 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

dimana:
- `pos` = posisi token (0, 1, 2, ...)
- `i` = indeks dimensi (0, 1, ..., d_model/2)
- `d_model` = dimensi embedding

**Cara kerja:** Vektor PE dijumlahkan langsung ke embedding token sebelum masuk ke attention layer.

```
x_input = TokenEmbedding(token) + PE(pos)
```

**Sifat penting:**
- Setiap posisi punya vektor unik dan deterministik
- Posisi berdekatan punya vektor yang mirip (smooth)
- Bisa generalisasi ke panjang sequence yang belum pernah dilihat saat training
- Tidak menambah parameter sama sekali

**Visualisasi pola frekuensi:**

```
Dimensi rendah (i kecil) → periode pendek → berubah cepat antar posisi
Dimensi tinggi (i besar) → periode panjang → berubah lambat antar posisi
```

Kombinasi keduanya membentuk "sidik jari" unik untuk setiap posisi, mirip sistem bilangan biner namun dalam domain kontinu.

---

#### Learned / Trainable Positional Encoding

Pendekatan ini memperlakukan posisi sebagai parameter yang dipelajari dari data, menggunakan `nn.Embedding(max_seq_len, d_model)`. Setiap posisi `0..max_len-1` punya vektor tersendiri yang dioptimasi via backprop.

```
x_input = TokenEmbedding(token) + PosEmbedding(pos_index)
```

**Kelebihan:** Model bisa belajar pola posisi yang paling berguna untuk task tertentu.

**Kekurangan:** Tidak bisa generalisasi ke panjang sequence di luar `max_len` training — berbeda dengan sinusoidal yang bisa ekstrapolasi.

**Dipakai oleh:** BERT (segment + position embedding), GPT-2, GPT-3.

---

#### RoPE: Rotary Position Embedding (Su et al., 2021)

RoPE adalah pendekatan yang **fundamentally berbeda** dari dua metode sebelumnya. Alih-alih menambahkan vektor posisi ke embedding, RoPE **memutar (rotate)** vektor Query dan Key di ruang kompleks berdasarkan posisi absolut mereka — sehingga dot product Q·K^T secara otomatis mengandung informasi posisi *relatif*.

**Intuisi geometris:**

Bayangkan setiap dimensi pasangan `(q_{2i}, q_{2i+1})` sebagai sebuah titik di bidang 2D. RoPE memutar titik itu dengan sudut yang proporsional terhadap posisi token:

```
θ_i = pos / 10000^(2i / d_k)

RoPE(q, pos)_{2i}   =  q_{2i}   · cos(θ_i) − q_{2i+1} · sin(θ_i)
RoPE(q, pos)_{2i+1} =  q_{2i}   · sin(θ_i) + q_{2i+1} · cos(θ_i)
```

Atau dalam notasi matriks rotasi per pasangan dimensi:

```
         ┌ cos(θ_i)  -sin(θ_i) ┐   ┌ q_{2i}   ┐
R(pos) · q = │                       │ · │           │
         └ sin(θ_i)   cos(θ_i) ┘   └ q_{2i+1} ┘
```

**Properti kunci — inner product hanya bergantung pada jarak relatif:**

Ini yang membuat RoPE istimewa. Jika token A di posisi `m` dan token B di posisi `n`:

```
⟨RoPE(q_m), RoPE(k_n)⟩  =  f(q, k, m − n)
```

Dot product antara Q dan K hanya bergantung pada **selisih posisi** `m − n`, bukan posisi absolut. Model otomatis belajar relasi posisional relatif.

**Perbandingan cara kerja:**

```
Sinusoidal / Learned:
  embedding → [+PE] → Q,K,V → Attention(Q·K^T)

RoPE:
  embedding → Q,K,V → [Rotate Q,K by pos] → Attention(Q_rot · K_rot^T)
                                               ↑ mengandung info posisi relatif
```

**Sifat penting RoPE:**
- Tidak menambah parameter (fixed rotation, bukan learned)
- Secara alami mendukung **posisi relatif** — lebih ekspresif dari absolute PE
- Long-context friendly: bisa di-scale ke konteks panjang (lihat YaRN, LongRoPE)
- Dipakai oleh hampir semua LLM modern: LLaMA, Mistral, Gemma, Qwen, dll.

---

### 1.4 Feed-Forward Network

Setiap layer transformer punya FFN yang diterapkan secara posisi-wise (independent per token):

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
```

atau dengan aktivasi GELU (lebih modern):

```
FFN(x) = GELU(x·W_1 + b_1) · W_2 + b_2
```

- `W_1 ∈ ℝ^(d_model × d_ff)`, biasanya `d_ff = 4 × d_model`
- `W_2 ∈ ℝ^(d_ff × d_model)`

FFN berfungsi sebagai "memori" yang menyimpan pengetahuan faktual, sementara attention menangani hubungan antar token.

---

### 1.5 Layer Normalization & Residual Connection

#### Residual Connection

```
output = LayerNorm(x + Sublayer(x))
```

Ini adalah "Pre-LN" (modern). Original paper menggunakan "Post-LN":

```
output = LayerNorm(x + Sublayer(x))   ← Post-LN (original)
output = x + Sublayer(LayerNorm(x))   ← Pre-LN (lebih stabil, umum dipakai)
```

Residual connection memungkinkan gradien mengalir langsung ke layer awal, mencegah vanishing gradient.

#### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

dimana:
- `μ = mean(x)` per token
- `σ = std(x)` per token
- `γ, β` = parameter yang dipelajari (scale dan shift)
- `ε` = konstanta kecil untuk numerik stability (misal `1e-5`)

---

## 2. Implementasi PyTorch From Scratch

> Semua implementasi hanya menggunakan komponen dasar: `nn.Linear`, `nn.Embedding`, `nn.Dropout`, operasi tensor standar.

### Setup Awal (Shared Components)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ① POSITIONAL ENCODING — SINUSOIDAL (Fixed)
# ============================================================
class SinusoidalPE(nn.Module):
    """
    Vektor posisi deterministik, dijumlahkan ke embedding.
    Tidak punya parameter yang dilatih.
    Cocok untuk: Encoder-Decoder klasik (Vaswani et al.)
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ============================================================
# ② POSITIONAL ENCODING — LEARNED / TRAINABLE
# ============================================================
class LearnedPE(nn.Module):
    """
    Vektor posisi dipelajari via backprop (nn.Embedding).
    Lebih fleksibel tapi tidak bisa ekstrapolasi ke luar max_len.
    Cocok untuk: BERT, GPT-2
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_emb(pos_ids))


# ============================================================
# ③ POSITIONAL ENCODING — RoPE (Rotary Position Embedding)
# ============================================================
class RotaryEmbedding(nn.Module):
    """
    Pra-hitung cos/sin cache untuk semua posisi.
    Tidak ada parameter yang dilatih.

    forward(q, k) → (q_rot, k_rot)
    Input : q, k  shape (batch, n_heads, seq_len, d_k)
    Output: q_rot, k_rot shape yang sama — siap dipakai di scaled_dot_product_attention.
    """
    def __init__(self, d_k, max_len=512, base=10000):
        super().__init__()
        # θ_i = 1 / base^(2i / d_k)  untuk i = 0..d_k/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)

        # Pra-hitung cos & sin untuk semua posisi sekaligus
        t    = torch.arange(max_len).float()
        freq = torch.outer(t, inv_freq)          # (max_len, d_k/2)
        emb  = torch.cat([freq, freq], dim=-1)   # (max_len, d_k)
        self.register_buffer('cos_cache', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cache', emb.sin().unsqueeze(0).unsqueeze(0))
        # Shape cache: (1, 1, max_len, d_k) → di-broadcast ke (batch, heads, seq, d_k)

    def _rotate_half(self, x):
        """Bantu rotasi: [x1..xn] → [-x_{n/2+1}..-xn, x1..x_{n/2}]"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        Terapkan rotasi posisi ke Q dan K.

        Args:
            q : (batch, n_heads, seq_len, d_k)
            k : (batch, n_heads, seq_len, d_k)
        Returns:
            q_rot, k_rot — shape sama dengan input
        """
        seq_len = q.size(2)
        cos = self.cos_cache[:, :, :seq_len, :]   # (1, 1, seq_len, d_k)
        sin = self.sin_cache[:, :, :seq_len, :]

        # Rumus: x_rot = x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ============================================================
# Scaled Dot-Product Attention (shared)
# ============================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: (batch, n_heads, seq_len, d_k)
    mask   : (batch, 1, seq_len, seq_len) atau (1, 1, seq_len, seq_len)
    """
    d_k    = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    return torch.matmul(F.softmax(scores, dim=-1), v)


# ============================================================
# Multi-Head Attention — mendukung ketiga jenis PE
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    pe_type: 'sinusoidal' | 'learned' | 'rope'
    
    Untuk sinusoidal & learned, PE sudah ditambahkan sebelum MHA,
    sehingga Q/K/V tidak perlu dimodifikasi di sini.
    
    Untuk rope, rotasi Q & K dilakukan di dalam forward() ini.
    """
    def __init__(self, d_model, n_heads, dropout=0.1,
                 pe_type='sinusoidal', max_len=512):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.pe_type  = pe_type

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # RoPE hanya butuh d_k (per head), bukan d_model
        if pe_type == 'rope':
            self.rope = RotaryEmbedding(self.d_k, max_len=max_len)

    def split_heads(self, x, B):
        return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        q = self.split_heads(self.W_q(query), B)   # (B, H, S, d_k)
        k = self.split_heads(self.W_k(key),   B)
        v = self.split_heads(self.W_v(value),  B)

        # Terapkan RoPE hanya jika diminta
        if self.pe_type == 'rope':
            q, k = self.rope(q, k)   # memanggil RotaryEmbedding.forward(q, k)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ============================================================
# Feed-Forward Network (shared)
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ============================================================
# Helper: Causal Mask
# ============================================================
def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


# ============================================================
# Perbandingan cepat ketiga PE
# ============================================================
if __name__ == "__main__":
    B, S, D = 2, 10, 64

    x = torch.randn(B, S, D)

    # ① Sinusoidal — tambah PE ke embedding
    sin_pe  = SinusoidalPE(D)
    x_sin   = sin_pe(x)
    print("Sinusoidal PE output:", x_sin.shape)  # (2, 10, 64)

    # ② Learned — tambah learnable PE ke embedding
    learn_pe = LearnedPE(D)
    x_learn  = learn_pe(x)
    print("Learned PE output   :", x_learn.shape)

    # ③ RoPE — rotasi Q & K di dalam MHA, embedding tidak dimodifikasi
    mha_rope = MultiHeadAttention(D, n_heads=4, pe_type='rope')
    x_rope   = mha_rope(x, x, x)    # query=key=value=x (self-attention)
    print("RoPE MHA output     :", x_rope.shape)  # (2, 10, 64)
```

---

### 2.1 Encoder Only

**Use case:** Klasifikasi teks, NER, sentence embedding (seperti BERT)

**PE yang cocok:** Sinusoidal atau Learned (RoPE bisa tapi jarang untuk encoder-only karena tidak sepenting di decoder)

```python
# ============================================================
# Encoder Layer
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 pe_type='sinusoidal', max_len=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout,
                                            pe_type=pe_type, max_len=max_len)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        nx = self.norm1(x)
        x  = x + self.drop(self.self_attn(nx, nx, nx, src_mask))
        x  = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ============================================================
# Encoder Only Model — pilih PE via pe_type
# ============================================================
class TransformerEncoderOnly(nn.Module):
    """
    pe_type: 'sinusoidal' | 'learned' | 'rope'
    
    Sinusoidal & Learned → PE dijumlahkan ke embedding sebelum masuk layer.
    RoPE               → tidak ada PE eksternal; rotasi terjadi di dalam MHA.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 d_ff=512, max_len=128, num_classes=2, dropout=0.1,
                 pe_type='sinusoidal'):
        super().__init__()
        self.pe_type   = pe_type
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale     = math.sqrt(d_model)

        # Pilih modul PE (hanya untuk sinusoidal & learned)
        if pe_type == 'sinusoidal':
            self.pos_enc = SinusoidalPE(d_model, max_len, dropout)
        elif pe_type == 'learned':
            self.pos_enc = LearnedPE(d_model, max_len, dropout)
        else:  # rope — tidak ada pe eksternal
            self.pos_enc = nn.Dropout(dropout)  # hanya dropout

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, pe_type, max_len)
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * self.scale
        x = self.pos_enc(x)   # no-op (dropout saja) untuk RoPE

        for layer in self.layers:
            x = layer(x, src_mask)

        x = self.norm(x)
        return self.classifier(x[:, 0, :])  # [CLS] token


# --- Contoh pemakaian: ketiga PE ---
if __name__ == "__main__":
    vocab_size, B, S = 1000, 2, 20

    for pe in ['sinusoidal', 'learned', 'rope']:
        model  = TransformerEncoderOnly(vocab_size=vocab_size, d_model=128,
                                        n_heads=4, n_layers=2, d_ff=256,
                                        num_classes=3, pe_type=pe)
        src    = torch.randint(1, vocab_size, (B, S))
        logits = model(src)
        n_par  = sum(p.numel() for p in model.parameters())
        print(f"[Encoder Only | PE={pe:12s}] output={logits.shape}  params={n_par:,}")
```

---

### 2.2 Decoder Only

**Use case:** Language modeling, text generation (seperti GPT, LLaMA)

**PE yang cocok:** Learned (GPT-2), RoPE (LLaMA/Mistral — ini standar modern)

```python
# ============================================================
# Decoder Layer (causal / masked self-attention)
# ============================================================
class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 pe_type='rope', max_len=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout,
                                            pe_type=pe_type, max_len=max_len)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        nx = self.norm1(x)
        x  = x + self.drop(self.self_attn(nx, nx, nx, causal_mask))
        x  = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ============================================================
# Decoder Only Model — pilih PE via pe_type
# ============================================================
class TransformerDecoderOnly(nn.Module):
    """
    pe_type: 'sinusoidal' | 'learned' | 'rope'
    
    Default 'rope' karena ini yang dipakai LLaMA, Mistral, dll.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 d_ff=512, max_len=256, dropout=0.1, pe_type='rope'):
        super().__init__()
        self.d_model  = d_model
        self.pe_type  = pe_type
        self.tok_emb  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale    = math.sqrt(d_model)

        # PE eksternal (untuk sinusoidal & learned)
        if pe_type == 'sinusoidal':
            self.pos_enc = SinusoidalPE(d_model, max_len, dropout)
        elif pe_type == 'learned':
            self.pos_enc = LearnedPE(d_model, max_len, dropout)
        else:  # rope
            self.pos_enc = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_heads, d_ff, dropout, pe_type, max_len)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, S   = idx.shape
        device = idx.device

        x = self.tok_emb(idx) * self.scale
        x = self.pos_enc(x)

        causal_mask = make_causal_mask(S, device)
        for layer in self.layers:
            x = layer(x, causal_mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=3):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -256:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if next_tok.item() == eos_id:
                break
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# --- Contoh pemakaian: ketiga PE ---
if __name__ == "__main__":
    vocab_size = 500

    for pe in ['sinusoidal', 'learned', 'rope']:
        model  = TransformerDecoderOnly(vocab_size=vocab_size, d_model=64,
                                        n_heads=4, n_layers=2, d_ff=256,
                                        max_len=64, pe_type=pe)
        idx    = torch.randint(1, vocab_size, (2, 10))
        tgt    = torch.randint(1, vocab_size, (2, 10))
        logits, loss = model(idx, tgt)
        n_par  = sum(p.numel() for p in model.parameters())
        print(f"[Decoder Only | PE={pe:12s}] logits={logits.shape}  loss={loss.item():.3f}  params={n_par:,}")
```

---

### 2.3 Encoder-Decoder

**Use case:** Machine translation, summarization (seperti T5, BART)

**PE yang cocok:** Sinusoidal (original T5 pakai relative bias, tapi sinusoidal cukup untuk pembelajaran), RoPE untuk versi modern

```python
# ============================================================
# Full Decoder Layer (dengan cross-attention ke encoder)
# ============================================================
class EncoderDecoderLayer(nn.Module):
    """
    1. Masked self-attention (causal) pada token decoder
    2. Cross-attention: Q dari decoder, K&V dari encoder output
    3. FFN
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 pe_type='sinusoidal', max_len=512):
        super().__init__()
        # Self-attention decoder (causal)
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout,
                                             pe_type=pe_type, max_len=max_len)
        # Cross-attention: Q dari decoder, K/V dari encoder
        # Cross-attention selalu sinusoidal/learned di sisi key-value
        # (encoder sudah encode posisi, RoPE di cross-attn tidak standar)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout,
                                             pe_type='sinusoidal', max_len=max_len)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention
        nx = self.norm1(x)
        x  = x + self.drop(self.self_attn(nx, nx, nx, tgt_mask))

        # 2. Cross-attention
        nx = self.norm2(x)
        x  = x + self.drop(self.cross_attn(nx, enc_output, enc_output, src_mask))

        # 3. FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ============================================================
# Encoder-Decoder Model — pilih PE via pe_type
# ============================================================
class TransformerEncoderDecoder(nn.Module):
    """
    pe_type berlaku untuk: encoder self-attn + decoder self-attn.
    Cross-attention selalu memakai sinusoidal (encoder output sudah encode posisi).
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=4,
                 n_enc_layers=3, n_dec_layers=3, d_ff=512, max_len=256, dropout=0.1,
                 pe_type='sinusoidal'):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type

        def make_pe():
            if pe_type == 'sinusoidal':
                return SinusoidalPE(d_model, max_len, dropout)
            elif pe_type == 'learned':
                return LearnedPE(d_model, max_len, dropout)
            else:
                return nn.Dropout(dropout)  # RoPE: no external PE

        # Encoder
        self.enc_emb    = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.enc_pe     = make_pe()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, pe_type, max_len)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder
        self.dec_emb    = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.dec_pe     = make_pe()
        self.dec_layers = nn.ModuleList([
            EncoderDecoderLayer(d_model, n_heads, d_ff, dropout, pe_type, max_len)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.scale = math.sqrt(d_model)

    def encode(self, src, src_mask=None):
        x = self.enc_emb(src) * self.scale
        x = self.enc_pe(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.dec_emb(tgt) * self.scale
        x = self.dec_pe(x)
        for layer in self.dec_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.dec_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)


# --- Contoh pemakaian: ketiga PE ---
if __name__ == "__main__":
    src_vocab, tgt_vocab = 1000, 800
    B, src_S, tgt_S = 2, 15, 10

    for pe in ['sinusoidal', 'learned', 'rope']:
        model = TransformerEncoderDecoder(src_vocab, tgt_vocab, d_model=64,
                                          n_heads=4, n_enc_layers=2, n_dec_layers=2,
                                          d_ff=256, pe_type=pe)
        src      = torch.randint(1, src_vocab, (B, src_S))
        tgt      = torch.randint(1, tgt_vocab, (B, tgt_S))
        tgt_mask = make_causal_mask(tgt_S, tgt.device)

        logits = model(src, tgt, tgt_mask=tgt_mask)
        n_par  = sum(p.numel() for p in model.parameters())
        print(f"[Enc-Dec | PE={pe:12s}] logits={logits.shape}  params={n_par:,}")
```

---

## 3. SLM Mini: Model Tanya Jawab Sederhana

> Arsitektur: Decoder-Only (GPT-style), dilatih untuk format `Q: <pertanyaan> A: <jawaban>`
> 
> **PE yang dipakai: RoPE** — pilihan standar untuk LLM/SLM modern.

### Struktur Lengkap

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter

# ── Semua kelas shared (SinusoidalPE, LearnedPE, RotaryEmbedding,
#    MultiHeadAttention, FeedForward, make_causal_mask,
#    DecoderOnlyLayer, TransformerDecoderOnly)
#    sudah didefinisikan di Bagian 2. ──


# ============================================================
# Tokenizer Sederhana
# ============================================================
class SimpleTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.PAD, self.UNK, self.BOS, self.EOS = 0, 1, 2, 3

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        self.word2id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2id:
                self.word2id[word] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        print(f"Vocabulary size: {len(self.word2id)}")

    def _tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def encode(self, text, add_special=True):
        ids = [self.word2id.get(t, self.UNK) for t in self._tokenize(text)]
        return ([self.BOS] + ids + [self.EOS]) if add_special else ids

    def decode(self, ids, skip_special=True):
        special = {self.PAD, self.BOS, self.EOS}
        return ' '.join(
            self.id2word.get(i, '<UNK>')
            for i in ids if not (skip_special and i in special)
        )

    @property
    def vocab_size(self):
        return len(self.word2id)


# ============================================================
# SLM Mini — Decoder-Only dengan RoPE
# ============================================================
class MiniSLM(nn.Module):
    """
    SLM mini untuk tanya jawab. Default pe_type='rope' (LLaMA-style).
    Bisa diganti ke 'sinusoidal' atau 'learned' untuk perbandingan.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=128, dropout=0.1, pe_type='rope'):
        super().__init__()
        self.d_model  = d_model
        self.max_len  = max_len
        self.pe_type  = pe_type

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale   = math.sqrt(d_model)

        if pe_type == 'sinusoidal':
            self.pos_enc = SinusoidalPE(d_model, max_len, dropout)
        elif pe_type == 'learned':
            self.pos_enc = LearnedPE(d_model, max_len, dropout)
        else:
            self.pos_enc = nn.Dropout(dropout)  # RoPE: rotasi di dalam MHA

        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_heads, d_ff, dropout, pe_type, max_len)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, S   = idx.shape
        device = idx.device
        if S > self.max_len:
            idx, S = idx[:, -self.max_len:], self.max_len

        x = self.tok_emb(idx) * self.scale
        x = self.pos_enc(x)

        mask = make_causal_mask(S, device)
        for layer in self.layers:
            x = layer(x, mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            if targets.size(1) > self.max_len:
                targets = targets[:, -self.max_len:]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8,
                 top_k=20, eos_id=3):
        self.eval()
        idx = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.max_len:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if next_tok.item() == eos_id:
                break
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ============================================================
# Dataset & Collate
# ============================================================
class QADataset(torch.utils.data.Dataset):
    def __init__(self, qa_pairs, tokenizer, max_len=128):
        self.samples = []
        for q, a in qa_pairs:
            ids = tokenizer.encode(f"q: {q} a: {a}", add_special=True)
            if len(ids) <= max_len:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return torch.tensor(ids[:-1], dtype=torch.long), \
               torch.tensor(ids[1:],  dtype=torch.long)


def collate_fn(batch):
    xs, ys  = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xp = torch.zeros(len(xs), max_len, dtype=torch.long)
    yp = torch.zeros(len(ys), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xp[i, :x.size(0)] = x
        yp[i, :y.size(0)] = y
    return xp, yp


# ============================================================
# Training Loop
# ============================================================
def train_slm(model, dataloader, optimizer, epochs=100, device='cpu'):
    model.train().to(device)
    for epoch in range(epochs):
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total/len(dataloader):.4f}")


# ============================================================
# Inference
# ============================================================
def answer_question(model, tokenizer, question, max_new_tokens=30,
                    temperature=0.7, device='cpu'):
    model.eval()
    prompt = f"q: {question} a:"
    ids    = tokenizer.encode(prompt, add_special=True)
    if ids[-1] == tokenizer.EOS:
        ids = ids[:-1]

    idx       = torch.tensor([ids], dtype=torch.long).to(device)
    generated = model.generate(idx, max_new_tokens, temperature, eos_id=tokenizer.EOS)
    full      = tokenizer.decode(generated[0].tolist())
    return full.split('a:')[-1].strip() if 'a:' in full else full


# ============================================================
# Main — bandingkan ketiga PE pada SLM
# ============================================================
if __name__ == "__main__":
    qa_data = [
        ("siapa kamu",           "saya adalah mini slm, model bahasa kecil"),
        ("apa itu transformer",  "transformer adalah arsitektur deep learning berbasis attention"),
        ("apa itu attention",    "attention adalah mekanisme yang memungkinkan model fokus pada bagian penting"),
        ("siapa pencipta transformer", "transformer diciptakan oleh google brain pada tahun 2017"),
        ("apa itu slm",          "slm adalah small language model, versi kecil dari llm"),
        ("apa itu llm",          "llm adalah large language model seperti gpt atau claude"),
        ("apa itu pytorch",      "pytorch adalah framework deep learning dari meta"),
        ("apa itu embedding",    "embedding adalah representasi vektor dari kata atau token"),
        ("apa itu tokenizer",    "tokenizer adalah alat untuk mengubah teks menjadi token"),
        ("apa itu gradient",     "gradient adalah turunan loss terhadap parameter model"),
        ("apa itu backprop",     "backpropagation adalah algoritma untuk menghitung gradient"),
        ("apa itu softmax",      "softmax adalah fungsi untuk mengubah skor menjadi probabilitas"),
        ("apa itu layer norm",   "layer normalization adalah teknik normalisasi per token"),
        ("apa itu fine tuning",  "fine tuning adalah proses melatih ulang model pada data spesifik"),
        ("apa itu dropout",      "dropout adalah teknik regularisasi dengan menonaktifkan neuron secara acak"),
        ("apa itu rope",         "rope adalah rotary position embedding yang memutar vektor query dan key"),
        ("apa itu residual",     "residual connection adalah jalur langsung dari input ke output layer"),
        ("apa itu cross entropy","cross entropy adalah fungsi loss untuk klasifikasi dan language modeling"),
        ("apa itu vocabulary",   "vocabulary adalah kumpulan semua kata yang diketahui oleh tokenizer"),
        ("apa itu generasi teks","generasi teks adalah proses model memprediksi token berikutnya secara berulang"),
    ]

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([f"q: {q} a: {a}" for q, a in qa_data])

    dataset    = QADataset(qa_data, tokenizer, max_len=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                             shuffle=True, collate_fn=collate_fn)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Latih & bandingkan ketiga PE ──
    for pe in ['sinusoidal', 'learned', 'rope']:
        print(f"\n{'='*50}")
        print(f"  Training MiniSLM dengan PE = {pe}")
        print('='*50)
        model = MiniSLM(vocab_size=tokenizer.vocab_size, d_model=64,
                        n_heads=4, n_layers=2, d_ff=128, max_len=64,
                        pe_type=pe)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"Total parameter: {n_par:,}\n")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        train_slm(model, dataloader, opt, epochs=100, device=device)

        print("\n─── Inference ───")
        for q in ["siapa kamu", "apa itu rope", "apa itu transformer"]:
            a = answer_question(model, tokenizer, q, device=device)
            print(f"Q: {q}\nA: {a}\n")
```


---

## Ringkasan Arsitektur

| Arsitektur | Attention | Use Case | Contoh Model |
|---|---|---|---|
| **Encoder Only** | Bidirectional (lihat semua arah) | Klasifikasi, NER, embedding | BERT, RoBERTa |
| **Decoder Only** | Causal (hanya ke kiri) | Text generation, chatbot | GPT, LLaMA, Claude |
| **Encoder-Decoder** | Bidirectional + Causal + Cross | Terjemahan, summarization | T5, BART, mT5 |

## Ringkasan Positional Encoding

| PE | Parameter | Cara kerja | Generalisasi | Dipakai oleh |
|---|---|---|---|---|
| **Sinusoidal** | 0 (fixed) | Tambah ke embedding | ✅ Bisa ekstrapolasi | Transformer asli |
| **Learned** | `max_len × d_model` | Tambah ke embedding | ❌ Hanya sampai `max_len` | BERT, GPT-2 |
| **RoPE** | 0 (fixed) | Rotasi Q & K di dalam attention | ✅ + Long-context scaling | LLaMA, Mistral, Gemma, Qwen |

**Kapan pilih mana:**
- Belajar / riset sederhana → **Sinusoidal** (paling mudah dipahami)
- Reproduksi BERT/GPT-2 → **Learned**
- Model modern / production → **RoPE** (terbaik untuk generalisasi posisi relatif)

## Tips Praktis

**Untuk eksperimen cepat:**
- Mulai dengan `d_model=64`, `n_heads=4`, `n_layers=2`
- `d_ff = 4 × d_model` adalah standar yang bagus
- Learning rate `3e-3` dengan AdamW + gradient clipping `1.0` biasanya aman

**Debugging:**
- Cek shape tensor di setiap tahap dengan `print(x.shape)`
- Pastikan loss turun (tidak NaN) setelah beberapa batch pertama
- Jika loss = NaN, kurangi learning rate atau periksa gradient clipping

**Scaling:**
- Tambah `n_layers` lebih efektif dari `d_model` untuk peningkatan kapasitas
- Weight tying (`lm_head.weight = tok_emb.weight`) mengurangi parameter ~20–30%
- Gunakan `torch.compile()` (PyTorch 2.0+) untuk speedup inference

---

*Referensi:*
- *"Attention Is All You Need" — Vaswani et al., 2017*
- *"RoFormer: Enhanced Transformer with Rotary Position Embedding" — Su et al., 2021*


