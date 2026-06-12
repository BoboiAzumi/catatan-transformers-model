# Tensor dari Scratch dengan TypeScript

> Panduan lengkap teori dan implementasi operasi tensor ala PyTorch, dibangun dari nol menggunakan TypeScript.

---

## Daftar Isi

1. [Torch Tensor — Struktur Dasar](#1-torch-tensor--struktur-dasar)
2. [`.view()` — Reshape Tensor](#2-view--reshape-tensor)
3. [`.transpose()` — Transpos Tensor](#3-transpose--transpos-tensor)
4. [`.backward()` — Autograd & Backpropagation](#4-backward--autograd--backpropagation)
   - [4b. `zeroGrad()` — Reset Gradien Seluruh Graf](#4b-zerograd--reset-gradien-seluruh-graf)
5. [`add()` dengan Broadcasting per Dimensi](#5-add-dengan-broadcasting-per-dimensi)
6. [`mul()` dengan Broadcasting per Dimensi](#6-mul-dengan-broadcasting-per-dimensi)
7. [`sub()` dengan Broadcasting per Dimensi](#7-sub-dengan-broadcasting-per-dimensi)
8. [`div()` dengan Broadcasting per Dimensi](#8-div-dengan-broadcasting-per-dimensi)
9. [`outer()` — Outer Product](#9-outer--outer-product)
10. [`matmul()` — Matrix Multiplication](#10-matmul--matrix-multiplication)
11. [`squeeze()` & `unsqueeze()` — Kelola Dimensi Tunggal](#11-squeeze--unsqueeze--kelola-dimensi-tunggal)
12. [`cat()` & `stack()` — Gabung Tensor](#12-cat--stack--gabung-tensor)
13. [`clamp()` — Pembatasan Nilai](#13-clamp--pembatasan-nilai)
14. [Fungsi Aktivasi — `sigmoid()`, `tanh()`, `softmax()`](#14-fungsi-aktivasi--sigmoid-tanh-softmax)
15. [`max()` & `min()` — Nilai Ekstrem & Argmax/Argmin](#15-max--min--nilai-ekstrem--argmaxargmin)
16. [`pow()` & `sqrt()` — Operasi Pangkat](#16-pow--sqrt--operasi-pangkat)
17. [`flatten()` & `permute()` — Reshape Lanjut](#17-flatten--permute--reshape-lanjut)
18. [Loss Functions — `mse_loss()`, `cross_entropy_loss()`](#18-loss-functions--mse_loss-cross_entropy_loss)
19. [`where()` & `masked_fill()` — Operasi Kondisional](#19-where--masked_fill--operasi-kondisional)
20. [`einsum()` — Einstein Summation](#20-einsum--einstein-summation)

---

## 1. Torch Tensor — Struktur Dasar

### Teori

Tensor adalah generalisasi dari skalar, vektor, dan matriks ke dimensi yang lebih tinggi:

| Rank | Nama     | Contoh Bentuk  |
|------|----------|----------------|
| 0    | Skalar   | `[]`           |
| 1    | Vektor   | `[3]`          |
| 2    | Matriks  | `[3, 4]`       |
| 3    | 3D Tensor| `[2, 3, 4]`    |
| N    | N-D Tensor | `[d1, d2, ...]`|

Sebuah tensor memiliki dua properti utama:
- **Shape** (`[d0, d1, ..., dN]`): dimensi di setiap sumbu.
- **Data** (flat array): nilai disimpan secara linear (row-major / C-order).

Untuk tensor shape `[2, 3]`, elemen `(i, j)` berada di indeks flat: `i * 3 + j`.

**Strides** adalah "langkah" (dalam jumlah elemen) yang harus dilewati untuk berpindah satu posisi pada setiap dimensi. Untuk shape `[d0, d1, ..., dN]`:

```
strides[N-1] = 1
strides[k]   = strides[k+1] * shape[k+1]
```

Contoh shape `[2, 3, 4]` → strides `[12, 4, 1]`.

### Implementasi

```typescript
// ============================================================
// tensor.ts — Implementasi Tensor dari Scratch
// ============================================================

class Tensor {
  data: Float32Array;   // penyimpanan data flat
  shape: number[];      // dimensi tensor
  strides: number[];    // stride per dimensi
  grad: Tensor | null = null;           // gradient (untuk autograd)
  requiresGrad: boolean = false;        // apakah perlu dihitung gradiennya
  _gradFn: (() => void) | null = null;  // fungsi backward
  _prevTensors: Tensor[] = [];          // tensor yang membentuk ini

  constructor(data: number[] | Float32Array, shape: number[]) {
    const totalSize = shape.reduce((acc, d) => acc * d, 1);
    if (data.length !== totalSize) {
      throw new Error(
        `Data length ${data.length} tidak cocok dengan shape ${shape} (total ${totalSize})`
      );
    }
    this.data   = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape  = [...shape];
    this.strides = Tensor.computeStrides(shape);
  }

  // ---- Utilitas Statis ----------------------------------------

  /** Hitung strides dari shape (C-order / row-major) */
  static computeStrides(shape: number[]): number[] {
    const strides = new Array<number>(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  /** Buat tensor berisi nol */
  static zeros(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape);
  }

  /** Buat tensor berisi satu */
  static ones(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size).fill(1), shape);
  }

  /** Buat tensor dari nested array */
  static fromArray(arr: any): Tensor {
    const shape: number[] = [];
    let cur = arr;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }
    const flat: number[] = [];
    const recurse = (a: any) => {
      if (Array.isArray(a)) a.forEach(recurse);
      else flat.push(a);
    };
    recurse(arr);
    return new Tensor(flat, shape);
  }

  // ---- Properti -----------------------------------------------

  get ndim(): number { return this.shape.length; }

  get size(): number { return this.data.length; }

  /** Konversi indeks multi-dimensi ke indeks flat */
  flatIndex(indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return idx;
  }

  /** Ambil nilai berdasarkan indeks multi-dimensi */
  get(...indices: number[]): number {
    return this.data[this.flatIndex(indices)];
  }

  /** Set nilai berdasarkan indeks multi-dimensi */
  set(value: number, ...indices: number[]): void {
    this.data[this.flatIndex(indices)] = value;
  }

  /** Tampilkan tensor secara ringkas */
  toString(): string {
    const formatRow = (data: Float32Array, offset: number, cols: number): string => {
      const row = Array.from(data.slice(offset, offset + cols))
        .map(v => v.toFixed(4));
      return `[${row.join(', ')}]`;
    };

    if (this.ndim === 1) {
      return formatRow(this.data, 0, this.shape[0]);
    }
    if (this.ndim === 2) {
      const [rows, cols] = this.shape;
      const lines = Array.from({ length: rows }, (_, i) =>
        `  ${formatRow(this.data, i * cols, cols)}`
      );
      return `[\n${lines.join(',\n')}\n]`;
    }
    return `Tensor(shape=${JSON.stringify(this.shape)}, data=[${Array.from(this.data).join(', ')}])`;
  }
}

// ---- Contoh Penggunaan ----------------------------------------

const t1 = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
console.log("Shape:", t1.shape);     // [2, 3]
console.log("Strides:", t1.strides); // [3, 1]
console.log("t1[0][1]:", t1.get(0, 1)); // 2
console.log(t1.toString());
// [
//   [1.0000, 2.0000, 3.0000],
//   [4.0000, 5.0000, 6.0000]
// ]

const t2 = Tensor.fromArray([[1, 2], [3, 4], [5, 6]]);
console.log("fromArray shape:", t2.shape); // [3, 2]
```

---

## 2. `.view()` — Reshape Tensor

### Teori

`.view()` (atau `.reshape()`) mengubah interpretasi shape tensor **tanpa menyalin data**. Elemen flat tetap sama, hanya cara "membacanya" yang berubah.

**Syarat:** total elemen harus sama.  
`d0 × d1 × ... = d0' × d1' × ...`

Satu dimensi boleh diisi `-1`, yang artinya "inferensi otomatis":

```
dim_infer = total_size / (perkalian dimensi lain)
```

Contoh: tensor shape `[2, 3, 4]` (24 elemen) dapat di-*view* menjadi `[6, 4]`, `[24]`, `[2, 12]`, `[4, -1]` → `[4, 6]`, dst.

Secara memori, `.view()` hanya mengubah `shape` dan `strides`, **bukan** `data`.

### Implementasi

```typescript
// Tambahkan method ke class Tensor

view(...newShape: number[]): Tensor {
  // Tangani nilai -1 (inferensi dimensi)
  const inferIdx = newShape.indexOf(-1);
  if (inferIdx !== -1) {
    const knownProduct = newShape
      .filter(d => d !== -1)
      .reduce((a, b) => a * b, 1);
    if (this.size % knownProduct !== 0) {
      throw new Error(`Tidak dapat infer dimensi -1: ${this.size} tidak habis dibagi ${knownProduct}`);
    }
    newShape = [...newShape];
    newShape[inferIdx] = this.size / knownProduct;
  }

  // Validasi total elemen
  const newSize = newShape.reduce((a, b) => a * b, 1);
  if (newSize !== this.size) {
    throw new Error(
      `view: total elemen tidak cocok. Shape asal ${this.shape} (${this.size}) → target ${newShape} (${newSize})`
    );
  }

  // Buat tensor baru yang berbagi data yang sama (zero-copy)
  const result = new Tensor(this.data, newShape);
  return result;
}

// ---- Contoh Penggunaan ----------------------------------------

const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 6]);
console.log("Asal:", a.shape);         // [2, 6]

const b = a.view(3, 4);
console.log("view(3,4):", b.shape);    // [3, 4]

const c = a.view(-1, 3);
console.log("view(-1,3):", c.shape);   // [4, 3]

const d = a.view(12);
console.log("view(12):", d.shape);     // [12]

// Zero-copy: data yang sama
b.data[0] = 99;
console.log("a[0][0] ikut berubah:", a.get(0, 0)); // 99
```

---

## 3. `.transpose()` — Transpos Tensor

### Teori

Transpos menukar dua dimensi (sumbu) dari sebuah tensor. Untuk matriks 2D, ini adalah transpos klasik:

```
A[i][j]  →  Aᵀ[j][i]
```

Untuk tensor N-D, kita hanya menukar dua sumbu tertentu `dim0` dan `dim1`.

**Trik efisiensi:** Alih-alih menyalin data, kita **menukar shape dan strides** dari dua dimensi tersebut. Hasilnya adalah *view* (non-contiguous) yang sama efisiennya.

Contoh: tensor shape `[2, 3]`, strides `[3, 1]`  
Setelah `transpose(0, 1)`: shape `[3, 2]`, strides `[1, 3]`.

Elemen `Aᵀ[j][i]` diakses di indeks flat `j * 1 + i * 3` = `i * 3 + j * 1`, persis sama dengan `A[i][j]`.

### Implementasi

```typescript
// Tambahkan method ke class Tensor

transpose(dim0: number, dim1: number): Tensor {
  const ndim = this.ndim;

  // Normalisasi indeks negatif (misal -1 = dimensi terakhir)
  if (dim0 < 0) dim0 += ndim;
  if (dim1 < 0) dim1 += ndim;

  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw new Error(`transpose: dimensi ${dim0} atau ${dim1} di luar range (ndim=${ndim})`);
  }

  // Salin shape dan strides, lalu tukar dua dimensi
  const newShape   = [...this.shape];
  const newStrides = [...this.strides];

  [newShape[dim0],   newShape[dim1]]   = [newShape[dim1],   newShape[dim0]];
  [newStrides[dim0], newStrides[dim1]] = [newStrides[dim1], newStrides[dim0]];

  // Buat tensor baru berbagi data yang sama
  const result    = new Tensor(this.data, newShape);
  result.strides  = newStrides;
  return result;
}

/** Shorthand untuk tensor 2D: .T */
get T(): Tensor {
  if (this.ndim !== 2) throw new Error(".T hanya untuk tensor 2D");
  return this.transpose(0, 1);
}

// ---- Contoh Penggunaan ----------------------------------------

const mat = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]); // shape [2, 3]
console.log("Asal:", mat.shape, "strides:", mat.strides); // [2,3], [3,1]

const matT = mat.T;
console.log("Transpos:", matT.shape, "strides:", matT.strides); // [3,2], [1,3]

// Verifikasi elemen
console.log("mat[0][2] =", mat.get(0, 2));   // 3
console.log("matT[2][0] =", matT.get(2, 0)); // 3  ← sama!

// Tensor 3D
const t3d = new Tensor(Array.from({length:24}, (_,i)=>i), [2,3,4]);
const t3dT = t3d.transpose(0, 2); // tukar sumbu 0 dan 2
console.log("3D transpose shape:", t3dT.shape); // [4,3,2]
```

---

## 4. `.backward()` — Autograd & Backpropagation

### Teori

**Autograd** (automatic differentiation) adalah mekanisme yang memungkinkan framework deep learning menghitung gradien secara otomatis menggunakan **chain rule**:

```
∂L/∂x = (∂L/∂z) × (∂z/∂x)
```

Di mana `z = f(x)` adalah operasi yang dilakukan.

Setiap operasi merekam:
1. **Input tensor** yang digunakan
2. **Fungsi backward** yang menjelaskan cara menghitung gradien input dari gradien output

**Computational Graph** dibangun secara dinamis (define-by-run), persis seperti PyTorch.

Contoh untuk `z = x + y`:
- Forward: `z.data = x.data + y.data`
- Backward: `∂L/∂x += ∂L/∂z`, `∂L/∂y += ∂L/∂z`

Contoh untuk `z = x * y`:
- Forward: `z.data = x.data * y.data`
- Backward: `∂L/∂x += ∂L/∂z * y.data`, `∂L/∂y += ∂L/∂z * x.data`

### Implementasi

```typescript
// Perbarui class Tensor dengan dukungan autograd

class Tensor {
  data: Float32Array;
  shape: number[];
  strides: number[];
  grad: Tensor | null = null;
  requiresGrad: boolean = false;
  _gradFn: (() => void) | null = null;
  _prevTensors: Tensor[] = [];

  // ... (constructor, computeStrides, dll. sama seperti sebelumnya)

  /** Tambahkan gradien (akumulasi, bukan assign) */
  private accumulateGrad(g: Tensor): void {
    if (this.grad === null) {
      this.grad = Tensor.zeros(this.shape);
    }
    for (let i = 0; i < this.data.length; i++) {
      this.grad.data[i] += g.data[i];
    }
  }

  /**
   * Jalankan backpropagation dari tensor ini.
   * Tensor ini harus berupa skalar (size = 1).
   */
  backward(gradient?: Tensor): void {
    if (this.size !== 1 && gradient === undefined) {
      throw new Error("backward() tanpa argumen hanya untuk tensor skalar");
    }

    // Inisialisasi gradien output = 1 (∂L/∂L = 1)
    const rootGrad = gradient ?? new Tensor([1], [1]);

    // Topological sort (DFS)
    const visited = new Set<Tensor>();
    const order: Tensor[] = [];

    const dfs = (node: Tensor) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const prev of node._prevTensors) dfs(prev);
      order.push(node);
    };
    dfs(this);

    // Inisialisasi grad di root
    this.grad = rootGrad;

    // Propagasi mundur dalam urutan terbalik
    for (const node of order.reverse()) {
      if (node._gradFn) node._gradFn();
    }
  }
}

// ---- Operasi yang mendukung Autograd --------------------------

/** Penjumlahan element-wise dengan dukungan backward */
function addWithGrad(a: Tensor, b: Tensor): Tensor {
  const outData = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) {
    outData[i] = a.data[i] + b.data[i];
  }
  const out = new Tensor(outData, [...a.shape]);
  out.requiresGrad = a.requiresGrad || b.requiresGrad;

  if (out.requiresGrad) {
    out._prevTensors = [a, b];
    out._gradFn = () => {
      // ∂L/∂a = ∂L/∂out * 1
      if (a.requiresGrad) {
        const ga = new Tensor(new Float32Array(out.grad!.data), a.shape);
        a.grad = a.grad
          ? new Tensor(a.grad.data.map((v, i) => v + ga.data[i]), a.shape)
          : ga;
      }
      // ∂L/∂b = ∂L/∂out * 1
      if (b.requiresGrad) {
        const gb = new Tensor(new Float32Array(out.grad!.data), b.shape);
        b.grad = b.grad
          ? new Tensor(b.grad.data.map((v, i) => v + gb.data[i]), b.shape)
          : gb;
      }
    };
  }
  return out;
}

/** Perkalian element-wise dengan dukungan backward */
function mulWithGrad(a: Tensor, b: Tensor): Tensor {
  const outData = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) {
    outData[i] = a.data[i] * b.data[i];
  }
  const out = new Tensor(outData, [...a.shape]);
  out.requiresGrad = a.requiresGrad || b.requiresGrad;

  if (out.requiresGrad) {
    const aData = new Float32Array(a.data); // simpan snapshot
    const bData = new Float32Array(b.data);
    out._prevTensors = [a, b];
    out._gradFn = () => {
      // ∂L/∂a = ∂L/∂out * b
      if (a.requiresGrad) {
        const ga = new Tensor(
          out.grad!.data.map((g, i) => g * bData[i]),
          a.shape
        );
        a.grad = a.grad
          ? new Tensor(a.grad.data.map((v, i) => v + ga.data[i]), a.shape)
          : ga;
      }
      // ∂L/∂b = ∂L/∂out * a
      if (b.requiresGrad) {
        const gb = new Tensor(
          out.grad!.data.map((g, i) => g * aData[i]),
          b.shape
        );
        b.grad = b.grad
          ? new Tensor(b.grad.data.map((v, i) => v + gb.data[i]), b.shape)
          : gb;
      }
    };
  }
  return out;
}

// ---- Contoh: Regresi Linear Sederhana -------------------------

// y = w*x + b, loss = (y - target)^2
const x      = new Tensor([2.0], [1]);
const target = new Tensor([5.0], [1]);
const w      = new Tensor([1.0], [1]); w.requiresGrad = true;
const b      = new Tensor([0.0], [1]); b.requiresGrad = true;

// Forward pass
const wx   = mulWithGrad(w, x);        // w*x
const y    = addWithGrad(wx, b);       // w*x + b
const diff = addWithGrad(
  y,
  new Tensor([-target.data[0]], [1])   // y - target
);
const loss = mulWithGrad(diff, diff);  // (y - target)^2

console.log("loss:", loss.data[0]);    // (1*2+0 - 5)^2 = 9

// Backward pass
loss.backward();

console.log("∂loss/∂w:", w.grad?.data[0]); // 2*(y-target)*x = 2*(-3)*2 = -12
console.log("∂loss/∂b:", b.grad?.data[0]); // 2*(y-target)*1 = 2*(-3)*1 = -6
```

---

## 4b. `zeroGrad()` — Reset Gradien Seluruh Graf

### Teori

Setiap kali `.backward()` dipanggil, gradien **diakumulasi** (`+=`) ke `.grad` masing-masing tensor. Ini disengaja di PyTorch untuk mendukung teknik seperti gradient accumulation. Namun akibatnya, **jika kita tidak mereset gradien sebelum iterasi berikutnya, nilainya akan terus menumpuk** dan menghasilkan update parameter yang salah.

Di PyTorch, solusinya adalah:

```python
optimizer.zero_grad()   # reset semua .grad ke None / nol
loss.backward()         # isi ulang .grad
optimizer.step()        # update parameter
```

Fungsi `zeroGrad` perlu **menelusuri seluruh computational graph** (bukan hanya node yang langsung terhubung ke loss), dan mereset `.grad` setiap tensor yang `requiresGrad == true` menjadi `null`.

**Dua strategi reset:**

| Strategi | Perilaku | Kapan digunakan |
|----------|----------|-----------------|
| Set ke `null` | `.grad` di-null-kan | Default PyTorch (`set_to_none=True`) |
| Set ke nol | `.grad` diisi `zeros` | Saat kode hilir cek `.grad` tanpa null guard |

**Kenapa harus menelusuri graf, bukan hanya reset array parameter?**  
Karena tensor intermediate (hasil `mul`, `add`, dll.) juga bisa menyimpan `.grad` jika `requiresGrad=true`. Kalau tidak direset, nilai stale tersebut bisa mempengaruhi iterasi berikutnya jika graph dipakai ulang.

### Visualisasi Masalah Tanpa `zeroGrad`

```
Iterasi 1:  w.grad = -12   ✓ benar
Iterasi 2:  w.grad = -12 + (-12) = -24   ✗ salah! (harusnya -12 lagi)
Iterasi 3:  w.grad = -36   ✗ terus menumpuk
```

### Implementasi

```typescript
/**
 * Telusuri seluruh computational graph mulai dari `root`
 * dan reset .grad semua tensor menjadi null (atau zeros).
 *
 * @param root       Tensor output (biasanya loss) sebagai titik awal traversal
 * @param setToNone  true  → set .grad = null  (default, lebih hemat memori)
 *                   false → set .grad = zeros  (kompatibel dengan kode yang cek .grad tanpa null guard)
 */
function zeroGrad(root: Tensor, setToNone: boolean = true): void {
  const visited = new Set<Tensor>();

  const traverse = (node: Tensor): void => {
    if (visited.has(node)) return; // hindari kunjungan ganda di graf bercabang
    visited.add(node);

    // Reset gradien node ini
    if (node.requiresGrad) {
      if (setToNone) {
        node.grad = null;
      } else {
        node.grad = Tensor.zeros(node.shape);
      }
    }

    // Lanjut ke semua tensor yang membentuk node ini
    for (const prev of node._prevTensors) {
      traverse(prev);
    }
  };

  traverse(root);
}

/**
 * Versi alternatif: reset gradien dari daftar parameter eksplisit.
 * Berguna ketika kita sudah tahu tensor mana saja yang merupakan parameter
 * (mirip pola `optimizer.zero_grad()` di PyTorch).
 *
 * @param params     Array tensor parameter yang ingin direset
 * @param setToNone  true → null, false → zeros
 */
function zeroGradParams(params: Tensor[], setToNone: boolean = true): void {
  for (const p of params) {
    if (setToNone) {
      p.grad = null;
    } else {
      p.grad = Tensor.zeros(p.shape);
    }
  }
}

// ---- Contoh: Training Loop Sederhana --------------------------

// Model: y = w*x + b
const xTrain  = new Tensor([2.0], [1]);
const yTarget = new Tensor([5.0], [1]);

const wParam = new Tensor([1.0], [1]); wParam.requiresGrad = true;
const bParam = new Tensor([0.0], [1]); bParam.requiresGrad = true;
const lr     = 0.01; // learning rate

console.log("=== Training loop 5 iterasi ===");

let lastLoss: Tensor = new Tensor([0], [1]);

for (let epoch = 0; epoch < 5; epoch++) {
  // --- 1. Forward pass ---
  const wxEp   = mulWithGrad(wParam, xTrain);
  const yPred  = addWithGrad(wxEp, bParam);
  const diffEp = addWithGrad(yPred, new Tensor([-yTarget.data[0]], [1]));
  const lossEp = mulWithGrad(diffEp, diffEp);
  lastLoss     = lossEp;

  // --- 2. Backward pass ---
  lossEp.backward();

  console.log(
    `Epoch ${epoch + 1} | loss=${lossEp.data[0].toFixed(4)}` +
    ` | ∂w=${wParam.grad?.data[0].toFixed(4)}` +
    ` | ∂b=${bParam.grad?.data[0].toFixed(4)}`
  );

  // --- 3. Update parameter (SGD manual) ---
  wParam.data[0] -= lr * (wParam.grad?.data[0] ?? 0);
  bParam.data[0] -= lr * (bParam.grad?.data[0] ?? 0);

  // --- 4. Reset gradien (WAJIB sebelum iterasi berikutnya) ---
  zeroGrad(lossEp);           // traversal otomatis lewat graph
  // atau: zeroGradParams([wParam, bParam]);  // manual, lebih efisien
}

// Output yang diharapkan (gradien tidak menumpuk, tiap epoch akurat):
// Epoch 1 | loss=9.0000 | ∂w=-12.0000 | ∂b=-6.0000
// Epoch 2 | loss=...    | ∂w=...      | ∂b=...     ← berbeda dari epoch 1, bukan ganda
// ...

// ---- Demonstrasi masalah TANPA zeroGrad ----------------------

console.log("\n=== TANPA zeroGrad (gradien menumpuk) ===");

const wBad = new Tensor([1.0], [1]); wBad.requiresGrad = true;
const bBad = new Tensor([0.0], [1]); bBad.requiresGrad = true;

for (let epoch = 0; epoch < 3; epoch++) {
  const wxB   = mulWithGrad(wBad, xTrain);
  const yB    = addWithGrad(wxB, bBad);
  const diffB = addWithGrad(yB, new Tensor([-yTarget.data[0]], [1]));
  const lossB = mulWithGrad(diffB, diffB);

  lossB.backward(); // TIDAK ada zeroGrad → grad terakumulasi!

  console.log(
    `Epoch ${epoch + 1} | ∂w=${wBad.grad?.data[0].toFixed(4)}` +
    ` | ∂b=${bBad.grad?.data[0].toFixed(4)}  ← menumpuk!`
  );
}
// Epoch 1 | ∂w=-12.0000 | ∂b=-6.0000
// Epoch 2 | ∂w=-24.0000 | ∂b=-12.0000  ← SALAH, harusnya ~-12 lagi
// Epoch 3 | ∂w=-36.0000 | ∂b=-18.0000  ← makin menyimpang

// ---- Demonstrasi setToNone vs zeros --------------------------

console.log("\n=== setToNone=true (default) ===");
const wTest = new Tensor([1.0], [1]); wTest.requiresGrad = true;
const bTest = new Tensor([0.0], [1]); bTest.requiresGrad = true;
const wxT   = mulWithGrad(wTest, xTrain);
const yT    = addWithGrad(wxT, bTest);
const diffT = addWithGrad(yT, new Tensor([-yTarget.data[0]], [1]));
const lossT = mulWithGrad(diffT, diffT);
lossT.backward();
console.log("Sebelum zero:", wTest.grad?.data[0]); // -12

zeroGrad(lossT, true);  // set ke null
console.log("Setelah zero (null):", wTest.grad);   // null

zeroGrad(lossT, false); // set ke zeros
console.log("Setelah zero (zeros):", wTest.grad?.data); // [0]
```

### Catatan Penting

**Kapan harus `zeroGrad`?**

```
for setiap batch:
    zeroGrad(loss) atau zeroGradParams(params)   ← sebelum forward
    loss = forward(x)
    loss.backward()
    optimizer.step()
```

**Perbedaan `zeroGrad(root)` vs `zeroGradParams(params)`:**

- `zeroGrad(root)` — traversal otomatis seluruh graph dari node loss. Lebih aman karena tidak ada parameter yang terlewat, namun sedikit lebih lambat karena harus DFS seluruh graph.
- `zeroGradParams(params)` — hanya reset tensor yang kita daftarkan secara eksplisit. Lebih cepat dan cocok bila parameter sudah dikelola dalam satu list (pola optimizer).

---

## 5. `add()` dengan Broadcasting per Dimensi

### Teori

**Broadcasting** mengizinkan operasi antara tensor dengan shape berbeda. Aturannya (seperti NumPy/PyTorch):

1. Shape disamakan dari belakang (right-align).
2. Dimensi dengan ukuran `1` dapat "diperluas" ke ukuran lain.
3. Jika ukuran berbeda dan keduanya bukan `1`, error.

Contoh:
```
A: [3, 4]   + B: [4]   → B di-broadcast menjadi [3, 4]
A: [3, 1]   + B: [1,4] → hasil [3, 4]
A: [2,3,4]  + B: [3,4] → B di-broadcast menjadi [2,3,4]
```

**Penjumlahan sepanjang dimensi** (seperti `torch.sum(x, dim=0)`):

```
x: [m, n]  →  sum(dim=0): [n]   (jumlahkan sepanjang baris)
x: [m, n]  →  sum(dim=1): [m]   (jumlahkan sepanjang kolom)
```

### Implementasi

```typescript
// ---- Broadcasting ----------------------------------------

/** Hitung output shape dari broadcasting dua shape */
function broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
  const result: number[] = [];
  const maxLen = Math.max(shapeA.length, shapeB.length);

  for (let i = 0; i < maxLen; i++) {
    const a = shapeA[shapeA.length - 1 - i] ?? 1;
    const b = shapeB[shapeB.length - 1 - i] ?? 1;
    if (a !== b && a !== 1 && b !== 1) {
      throw new Error(`Shape tidak kompatibel untuk broadcasting: ${shapeA} vs ${shapeB}`);
    }
    result.unshift(Math.max(a, b));
  }
  return result;
}

/** Ambil elemen dari tensor dengan mempertimbangkan broadcasting */
function getWithBroadcast(tensor: Tensor, outShape: number[], flatIdx: number): number {
  // Konversi flatIdx ke multi-index di outShape
  const multiIdx: number[] = new Array(outShape.length);
  let rem = flatIdx;
  for (let d = outShape.length - 1; d >= 0; d--) {
    multiIdx[d] = rem % outShape[d];
    rem = Math.floor(rem / outShape[d]);
  }

  // Sesuaikan ke shape tensor (padding kiri dengan 1)
  const ndim = tensor.shape.length;
  const offset = outShape.length - ndim;
  const tensorIdx: number[] = new Array(ndim);
  for (let d = 0; d < ndim; d++) {
    const outD = multiIdx[d + offset];
    tensorIdx[d] = tensor.shape[d] === 1 ? 0 : outD; // broadcast: selalu ambil index 0 jika dim=1
  }

  return tensor.get(...tensorIdx);
}

/** Penjumlahan dengan broadcasting */
function add(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize  = outShape.reduce((x, y) => x * y, 1);
  const outData  = new Float32Array(outSize);

  for (let i = 0; i < outSize; i++) {
    outData[i] = getWithBroadcast(a, outShape, i)
               + getWithBroadcast(b, outShape, i);
  }
  return new Tensor(outData, outShape);
}

/** Sum sepanjang satu dimensi */
function sumDim(tensor: Tensor, dim: number, keepdim: boolean = false): Tensor {
  if (dim < 0) dim += tensor.ndim;

  const outShape = [...tensor.shape];
  outShape[dim]  = 1;

  const outSize  = outShape.reduce((a, b) => a * b, 1);
  const outData  = new Float32Array(outSize);

  for (let i = 0; i < tensor.size; i++) {
    // Konversi flat i ke multi-index
    const multiIdx: number[] = new Array(tensor.ndim);
    let rem = i;
    for (let d = tensor.ndim - 1; d >= 0; d--) {
      multiIdx[d] = rem % tensor.shape[d];
      rem = Math.floor(rem / tensor.shape[d]);
    }
    // Indeks output: set dim ke 0
    const outIdx = [...multiIdx];
    outIdx[dim]  = 0;
    const outFlat = outIdx.reduce((acc, idx, d) => acc * outShape[d] + idx, 0);
    // Ganti dengan formula langsung:
    let outFlat2 = 0;
    let stride   = 1;
    for (let d = outShape.length - 1; d >= 0; d--) {
      outFlat2 += outIdx[d] * stride;
      stride   *= outShape[d];
    }
    outData[outFlat2] += tensor.data[i];
  }

  if (keepdim) {
    return new Tensor(outData, outShape);
  }
  // Hapus dimensi yang sudah di-reduce
  const finalShape = tensor.shape.filter((_, d) => d !== dim);
  return new Tensor(outData, finalShape.length > 0 ? finalShape : [1]);
}

// ---- Contoh Penggunaan ----------------------------------------

// Broadcasting: [3, 4] + [4]
const A = new Tensor(Array.from({length:12}, (_,i) => i+1), [3, 4]);
const v = new Tensor([10, 20, 30, 40], [4]);
const C = add(A, v);
console.log("add broadcast [3,4]+[4]:", C.shape); // [3, 4]
console.log(C.toString());
// Baris 0: [11, 22, 33, 44]
// Baris 1: [15, 26, 37, 48]
// Baris 2: [19, 30, 41, 52]

// Sum sepanjang dim=0
const s0 = sumDim(A, 0);
console.log("sum dim=0:", s0.shape, s0.data); // [4]: [21,24,27,30] + 3+...

// Sum sepanjang dim=1
const s1 = sumDim(A, 1);
console.log("sum dim=1:", s1.shape, s1.data); // [3]: [10,26,42]
```

---

## 6. `mul()` dengan Broadcasting per Dimensi

### Teori

Perkalian element-wise (**Hadamard product**) antara dua tensor:

```
C[i][j] = A[i][j] × B[i][j]
```

Aturan broadcasting-nya identik dengan `add()`. Perbedaannya hanya pada operasinya: penjumlahan diganti perkalian.

**Product sepanjang dimensi** (`torch.prod(x, dim=k)`): mengalikan semua elemen di sepanjang dimensi tertentu.

Contoh:
```
x = [[1, 2, 3],    prod(dim=1) = [6, 120]
     [4, 5, 6]]
```

### Implementasi

```typescript
/** Perkalian element-wise dengan broadcasting */
function mul(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize  = outShape.reduce((x, y) => x * y, 1);
  const outData  = new Float32Array(outSize);

  for (let i = 0; i < outSize; i++) {
    outData[i] = getWithBroadcast(a, outShape, i)
               * getWithBroadcast(b, outShape, i);
  }
  return new Tensor(outData, outShape);
}

/** Product sepanjang satu dimensi */
function prodDim(tensor: Tensor, dim: number, keepdim: boolean = false): Tensor {
  if (dim < 0) dim += tensor.ndim;

  const outShape = [...tensor.shape];
  outShape[dim]  = 1;
  const outSize  = outShape.reduce((a, b) => a * b, 1);
  const outData  = new Float32Array(outSize).fill(1); // init 1 untuk perkalian

  for (let i = 0; i < tensor.size; i++) {
    const multiIdx: number[] = new Array(tensor.ndim);
    let rem = i;
    for (let d = tensor.ndim - 1; d >= 0; d--) {
      multiIdx[d] = rem % tensor.shape[d];
      rem = Math.floor(rem / tensor.shape[d]);
    }
    const outIdx = [...multiIdx];
    outIdx[dim]  = 0;

    let outFlat = 0;
    let stride  = 1;
    for (let d = outShape.length - 1; d >= 0; d--) {
      outFlat += outIdx[d] * stride;
      stride  *= outShape[d];
    }
    outData[outFlat] *= tensor.data[i];
  }

  if (keepdim) return new Tensor(outData, outShape);
  const finalShape = tensor.shape.filter((_, d) => d !== dim);
  return new Tensor(outData, finalShape.length > 0 ? finalShape : [1]);
}

// ---- Contoh Penggunaan ----------------------------------------

// Perkalian scalar broadcasting: [3, 4] * [1]
const M   = new Tensor(Array.from({length: 6}, (_, i) => i + 1), [2, 3]);
const scl = new Tensor([2], [1]);
const Mscaled = mul(M, scl);
console.log("mul scalar:", Mscaled.data); // [2,4,6,8,10,12]

// Perkalian per baris: [2, 3] * [2, 1]
const rowWeights = new Tensor([10, 100], [2, 1]);
const Mweighted  = mul(M, rowWeights);
console.log("mul row-wise:", Mweighted.data); // [10,20,30,100,200,300]

// Product sepanjang dim=1
const P = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
const p1 = prodDim(P, 1);
console.log("prod dim=1:", p1.data); // [6, 120]
```

---

## 7. `sub()` dengan Broadcasting per Dimensi

### Teori

Pengurangan element-wise:

```
C[i][j] = A[i][j] - B[i][j]
```

Secara matematis, `A - B = A + (-B)`, sehingga aturan broadcasting-nya sama persis.

**Pengurangan dengan mean** (centering data, sering digunakan dalam normalisasi):
```
X_centered = X - mean(X, dim=k)
```
Ini adalah pola umum di layer normalization dan batch normalization.

### Implementasi

```typescript
/** Pengurangan element-wise dengan broadcasting */
function sub(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize  = outShape.reduce((x, y) => x * y, 1);
  const outData  = new Float32Array(outSize);

  for (let i = 0; i < outSize; i++) {
    outData[i] = getWithBroadcast(a, outShape, i)
               - getWithBroadcast(b, outShape, i);
  }
  return new Tensor(outData, outShape);
}

/** Mean sepanjang satu dimensi */
function meanDim(tensor: Tensor, dim: number, keepdim: boolean = true): Tensor {
  const sumT = sumDim(tensor, dim, keepdim);
  const n    = tensor.shape[dim < 0 ? dim + tensor.ndim : dim];
  const out  = new Float32Array(sumT.data.length);
  for (let i = 0; i < sumT.data.length; i++) {
    out[i] = sumT.data[i] / n;
  }
  return new Tensor(out, sumT.shape);
}

/** Centering: X - mean(X, dim) */
function center(tensor: Tensor, dim: number): Tensor {
  const mu = meanDim(tensor, dim, true); // keepdim=true agar bisa broadcast
  return sub(tensor, mu);
}

// ---- Contoh Penggunaan ----------------------------------------

// Sub broadcasting: [3] - skalar
const x2 = new Tensor([5, 10, 15], [3]);
const y2 = new Tensor([3], [1]);
console.log("sub broadcast:", sub(x2, y2).data); // [2, 7, 12]

// Centering tiap kolom pada matriks [3, 4]
const X = new Tensor([
  2, 4, 1, 8,
  6, 2, 3, 2,
  4, 6, 5, 4
], [3, 4]);

const Xc = center(X, 0); // kurangi mean tiap kolom
console.log("X centered shape:", Xc.shape); // [3, 4]
// Verifikasi: setiap kolom harus berjumlah ~0
const colSum = sumDim(Xc, 0);
console.log("Col sums (harus ~0):", colSum.data);

// Sub dua matriks
const A2 = new Tensor([1,2,3,4,5,6], [2,3]);
const B2 = new Tensor([6,5,4,3,2,1], [2,3]);
console.log("A - B:", sub(A2, B2).data); // [-5,-3,-1,1,3,5]
```

---

## 8. `div()` dengan Broadcasting per Dimensi

### Teori

Pembagian element-wise:

```
C[i][j] = A[i][j] / B[i][j]
```

Pola umum yang paling sering digunakan adalah **normalisasi**:

```
X_norm = (X - μ) / σ
```

Di mana `σ` (std deviasi) sering di-broadcast dari shape `[1, n]` ke `[m, n]`.

**Perhatian numerik:** Selalu tambahkan `ε` (epsilon, misal `1e-8`) pada penyebut untuk menghindari pembagian oleh nol.

```
X_norm = X / (std(X) + ε)
```

### Implementasi

```typescript
/** Pembagian element-wise dengan broadcasting */
function div(a: Tensor, b: Tensor, eps: number = 0): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize  = outShape.reduce((x, y) => x * y, 1);
  const outData  = new Float32Array(outSize);

  for (let i = 0; i < outSize; i++) {
    const bVal = getWithBroadcast(b, outShape, i) + eps;
    if (bVal === 0) throw new Error(`Pembagian oleh nol pada indeks ${i}`);
    outData[i] = getWithBroadcast(a, outShape, i) / bVal;
  }
  return new Tensor(outData, outShape);
}

/** Variance sepanjang satu dimensi */
function varDim(tensor: Tensor, dim: number, keepdim: boolean = true): Tensor {
  const mu  = meanDim(tensor, dim, keepdim);        // mean
  const diff = sub(tensor, mu);                     // X - μ
  const sq   = mul(diff, diff);                     // (X - μ)²
  return meanDim(sq, dim, keepdim);                 // mean of squares = variance
}

/** Std sepanjang satu dimensi */
function stdDim(tensor: Tensor, dim: number, keepdim: boolean = true): Tensor {
  const variance = varDim(tensor, dim, keepdim);
  const out = new Float32Array(variance.data.length);
  for (let i = 0; i < variance.data.length; i++) {
    out[i] = Math.sqrt(variance.data[i]);
  }
  return new Tensor(out, variance.shape);
}

/** Normalisasi Z-score per dimensi */
function normalize(tensor: Tensor, dim: number, eps: number = 1e-8): Tensor {
  const mu  = meanDim(tensor, dim, true);
  const std = stdDim(tensor, dim, true);
  const centered = sub(tensor, mu);
  return div(centered, std, eps);
}

// ---- Contoh Penggunaan ----------------------------------------

// Pembagian sederhana dengan scalar
const scores = new Tensor([10, 20, 30, 40], [4]);
const scale  = new Tensor([10], [1]);
console.log("div scalar:", div(scores, scale).data); // [1, 2, 3, 4]

// Pembagian per kolom: softmax denominator
const logits = new Tensor([
  1.0, 2.0, 0.5,
  3.0, 0.1, 1.5
], [2, 3]);

// Hitung exp terlebih dahulu
const expData = new Float32Array(logits.data.map(Math.exp));
const expT    = new Tensor(expData, logits.shape);

// Sum sepanjang dim=1 (keepdim)
const expSum  = sumDim(expT, 1, true); // shape [2,1]
const softmax = div(expT, expSum);
console.log("softmax shape:", softmax.shape); // [2, 3]
console.log("softmax row 0 sum:", softmax.data.slice(0,3).reduce((a,b)=>a+b,0)); // ~1.0

// Normalisasi Z-score
const X3 = new Tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], [2, 3]);
const Xnorm = normalize(X3, 1);
console.log("Z-normalized:", Xnorm.data);
// Tiap baris mendekati mean=0, std=1
```

---

## 9. `outer()` — Outer Product

### Teori

**Outer product** (produk luar) antara dua vektor menghasilkan matriks:

```
c[i][j] = a[i] × b[j]
```

Untuk vektor `a` berukuran `m` dan `b` berukuran `n`, hasilnya adalah matriks `[m, n]`.

Secara geometris, outer product menghasilkan semua kombinasi pasangan elemen dari dua vektor.

**Generalisasi untuk tensor N-D:** PyTorch memungkinkan outer product antara dua vektor menjadi matriks 2D. Versi umum (tensor product) dapat menghasilkan tensor dengan rank lebih tinggi.

**Hubungan dengan matrix multiplication:**
```
outer(a, b) = a.reshape(-1, 1) @ b.reshape(1, -1)
             = column_vector × row_vector
```

**Aplikasi penting:**
- Attention matrix: `scores = outer(query, key)` (disederhanakan)
- Covariance matrix: `C = outer(x - μ, x - μ)`
- Rank-1 update dalam optimasi

### Implementasi

```typescript
/** Outer product dua vektor → matriks [m, n] */
function outer(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 1) throw new Error(`outer: a harus vektor 1D, dapat shape ${a.shape}`);
  if (b.ndim !== 1) throw new Error(`outer: b harus vektor 1D, dapat shape ${b.shape}`);

  const m   = a.shape[0];
  const n   = b.shape[0];
  const out = new Float32Array(m * n);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      out[i * n + j] = a.data[i] * b.data[j];
    }
  }
  return new Tensor(out, [m, n]);
}

/** Outer product untuk tensor N-D (flatten dulu, lalu outer) */
function outerND(a: Tensor, b: Tensor): Tensor {
  const aFlat = a.view(a.size);
  const bFlat = b.view(b.size);
  return outer(aFlat, bFlat);
}

/** Batch outer product: [batch, m] × [batch, n] → [batch, m, n] */
function batchOuter(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error("batchOuter: kedua input harus 2D [batch, dim]");
  }
  if (a.shape[0] !== b.shape[0]) {
    throw new Error(`batchOuter: batch size tidak sama: ${a.shape[0]} vs ${b.shape[0]}`);
  }

  const [batch, m] = a.shape;
  const n   = b.shape[1];
  const out = new Float32Array(batch * m * n);

  for (let k = 0; k < batch; k++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        out[k * m * n + i * n + j] = a.data[k * m + i] * b.data[k * n + j];
      }
    }
  }
  return new Tensor(out, [batch, m, n]);
}

// ---- Contoh Penggunaan ----------------------------------------

const u = new Tensor([1, 2, 3], [3]);
const v3 = new Tensor([10, 20, 30, 40], [4]);

const O = outer(u, v3);
console.log("outer([3],[4]) shape:", O.shape); // [3, 4]
console.log(O.toString());
// [
//   [10, 20, 30, 40],
//   [20, 40, 60, 80],
//   [30, 60, 90, 120]
// ]

// Covariance: outer(x, x)
const x4 = new Tensor([1, 2, 3], [3]);
const cov = outer(x4, x4);
console.log("Covariance-like outer:", cov.toString());

// Batch outer product: 2 pasang vektor sekaligus
const queries = new Tensor([1, 0, 0, 1], [2, 2]); // 2 query vektor
const keys    = new Tensor([1, 2, 3, 0, 1, 0], [2, 3]); // 2 key vektor
const attn    = batchOuter(queries, keys);
console.log("Batch outer (attention):", attn.shape); // [2, 2, 3]
```

---

## Ringkasan: Semua Operasi dalam Satu File

```typescript
// ============================================================
// tensor_complete.ts — Implementasi lengkap semua operasi
// ============================================================

// Jalankan: npx ts-node tensor_complete.ts

class Tensor {
  data: Float32Array;
  shape: number[];
  strides: number[];
  grad: Tensor | null = null;
  requiresGrad: boolean = false;
  _gradFn: (() => void) | null = null;
  _prevTensors: Tensor[] = [];

  constructor(data: number[] | Float32Array, shape: number[]) {
    const total = shape.reduce((a, b) => a * b, 1);
    if (data.length !== total)
      throw new Error(`Data length ${data.length} ≠ shape total ${total}`);
    this.data    = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape   = [...shape];
    this.strides = Tensor.computeStrides(shape);
  }

  static computeStrides(shape: number[]): number[] {
    const s = new Array<number>(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      s[i] = stride;
      stride *= shape[i];
    }
    return s;
  }

  static zeros(shape: number[]) {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }

  static ones(shape: number[]) {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1), shape);
  }

  get ndim() { return this.shape.length; }
  get size()  { return this.data.length; }

  get(...idx: number[]) {
    return this.data[idx.reduce((acc, i, d) => acc + i * this.strides[d], 0)];
  }

  view(...newShape: number[]): Tensor {
    const inf = newShape.indexOf(-1);
    if (inf !== -1) {
      const known = newShape.filter(d => d !== -1).reduce((a, b) => a * b, 1);
      newShape    = [...newShape];
      newShape[inf] = this.size / known;
    }
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) throw new Error(`view: size mismatch ${this.size} vs ${newSize}`);
    return new Tensor(this.data, newShape);
  }

  transpose(d0: number, d1: number): Tensor {
    const ns = [...this.shape];
    const nst = [...this.strides];
    [ns[d0], ns[d1]]   = [ns[d1], ns[d0]];
    [nst[d0], nst[d1]] = [nst[d1], nst[d0]];
    const r     = new Tensor(this.data, ns);
    r.strides   = nst;
    return r;
  }

  get T() { return this.transpose(0, 1); }

  backward(gradient?: Tensor) {
    this.grad = gradient ?? new Tensor([1], [1]);
    const visited = new Set<Tensor>(), order: Tensor[] = [];
    const dfs = (n: Tensor) => {
      if (visited.has(n)) return;
      visited.add(n);
      n._prevTensors.forEach(dfs);
      order.push(n);
    };
    dfs(this);
    for (const node of order.reverse())
      if (node._gradFn) node._gradFn();
  }
}

function broadcastShapes(sA: number[], sB: number[]): number[] {
  const res: number[] = [];
  for (let i = 0; i < Math.max(sA.length, sB.length); i++) {
    const a = sA[sA.length - 1 - i] ?? 1;
    const b = sB[sB.length - 1 - i] ?? 1;
    if (a !== b && a !== 1 && b !== 1) throw new Error(`Broadcast error: ${sA} vs ${sB}`);
    res.unshift(Math.max(a, b));
  }
  return res;
}

function getBcast(t: Tensor, outShape: number[], flat: number): number {
  const mi: number[] = [], ndim = outShape.length;
  let rem = flat;
  for (let d = ndim - 1; d >= 0; d--) {
    mi[d] = rem % outShape[d];
    rem   = Math.floor(rem / outShape[d]);
  }
  const off = ndim - t.ndim;
  return t.get(...Array.from({length: t.ndim}, (_, d) =>
    t.shape[d] === 1 ? 0 : mi[d + off]
  ));
}

function elemwise(a: Tensor, b: Tensor, fn: (x: number, y: number) => number): Tensor {
  const os = broadcastShapes(a.shape, b.shape);
  const sz = os.reduce((x, y) => x * y, 1);
  const d  = new Float32Array(sz);
  for (let i = 0; i < sz; i++) d[i] = fn(getBcast(a, os, i), getBcast(b, os, i));
  return new Tensor(d, os);
}

const add  = (a: Tensor, b: Tensor) => elemwise(a, b, (x, y) => x + y);
const mul  = (a: Tensor, b: Tensor) => elemwise(a, b, (x, y) => x * y);
const sub  = (a: Tensor, b: Tensor) => elemwise(a, b, (x, y) => x - y);
const div  = (a: Tensor, b: Tensor, eps = 0) => elemwise(a, b, (x, y) => x / (y + eps));

function sumDim(t: Tensor, dim: number, keepdim = false): Tensor {
  if (dim < 0) dim += t.ndim;
  const os = [...t.shape]; os[dim] = 1;
  const od = new Float32Array(os.reduce((a, b) => a * b, 1));
  for (let i = 0; i < t.size; i++) {
    let rem = i, of2 = 0, str = 1;
    const mi = Array.from({length: t.ndim}, (_, d) => {
      const idx = Math.floor(rem / t.strides[d]) % t.shape[d];
      return idx;
    });
    mi[dim] = 0;
    for (let d = os.length - 1; d >= 0; d--) { of2 += mi[d] * str; str *= os[d]; }
    od[of2] += t.data[i];
  }
  if (keepdim) return new Tensor(od, os);
  return new Tensor(od, t.shape.filter((_, d) => d !== dim) || [1]);
}

function meanDim(t: Tensor, dim: number, keepdim = true): Tensor {
  const s  = sumDim(t, dim, keepdim);
  const n  = t.shape[dim < 0 ? dim + t.ndim : dim];
  return new Tensor(s.data.map(v => v / n), s.shape);
}

function outer(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 1 || b.ndim !== 1) throw new Error("outer: perlu dua vektor 1D");
  const [m, n] = [a.shape[0], b.shape[0]];
  const d = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      d[i * n + j] = a.data[i] * b.data[j];
  return new Tensor(d, [m, n]);
}

// ---- Demo Cepat -----------------------------------------------

const t = new Tensor([1,2,3,4,5,6], [2,3]);
console.log("=== view [3,2] ===");
console.log(t.view(3, 2).shape);

console.log("\n=== transpose ===");
console.log(t.T.shape);           // [3, 2]
console.log(t.T.get(0, 1));       // t[1][0] = 4

console.log("\n=== add broadcast [2,3]+[3] ===");
const bias = new Tensor([1, 2, 3], [3]);
console.log(add(t, bias).data);   // [2,4,6,5,7,9]

console.log("\n=== outer [3]×[4] ===");
const o = outer(new Tensor([1,2,3],[3]), new Tensor([1,2,3,4],[4]));
console.log(o.shape);             // [3, 4]

console.log("\n=== backward: z = x*y + x ===");
const x = new Tensor([3], [1]); x.requiresGrad = true;
const y = new Tensor([2], [1]); y.requiresGrad = true;
// z = x*y
const xy = mul(x, y);
xy.requiresGrad = true;
// Setup gradFn manual sederhana untuk demo
xy._prevTensors = [x, y];
const xSnap = new Float32Array(x.data), ySnap = new Float32Array(y.data);
xy._gradFn = () => {
  x.grad = new Tensor([xy.grad!.data[0] * ySnap[0]], [1]);
  y.grad = new Tensor([xy.grad!.data[0] * xSnap[0]], [1]);
};
xy.backward();
console.log("∂(x*y)/∂x =", x.grad?.data[0]); // 2
console.log("∂(x*y)/∂y =", y.grad?.data[0]); // 3
```

---

---

## 10. `matmul()` — Matrix Multiplication

### Teori

Perkalian matriks adalah operasi paling fundamental dalam deep learning. Untuk dua matriks `A [m, k]` dan `B [k, n]`:

```
C[i][j] = Σ_p  A[i][p] × B[p][j]
```

Syarat: dimensi **dalam** harus sama (`k`). Hasil: `[m, n]`.

**Kasus umum:**

| Input A | Input B | Output | Nama |
|---------|---------|--------|------|
| `[m, k]` | `[k, n]` | `[m, n]` | Matrix × Matrix |
| `[k]` | `[k]` | skalar | Dot product |
| `[m, k]` | `[k]` | `[m]` | Matrix × Vektor |
| `[batch, m, k]` | `[batch, k, n]` | `[batch, m, n]` | Batch matmul |

**Backward pass:**
```
∂L/∂A = ∂L/∂C  @  Bᵀ      → shape [m,k]
∂L/∂B = Aᵀ     @  ∂L/∂C   → shape [k,n]
```

### Implementasi

```typescript
/** Matrix multiplication: [m,k] @ [k,n] → [m,n] */
function matmul(A: Tensor, B: Tensor): Tensor {
  // Tangani vektor 1D → 2D sementara
  const aIs1D = A.ndim === 1;
  const bIs1D = B.ndim === 1;
  const a2 = aIs1D ? A.view(1, A.shape[0]) : A;
  const b2 = bIs1D ? B.view(B.shape[0], 1) : B;

  if (a2.ndim !== 2 || b2.ndim !== 2) {
    throw new Error(`matmul: perlu tensor 2D, dapat ${A.shape} dan ${B.shape}`);
  }

  const [m, k] = a2.shape;
  const [k2, n] = b2.shape;
  if (k !== k2) {
    throw new Error(`matmul: inner dim tidak cocok: ${k} vs ${k2}`);
  }

  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += a2.get(i, p) * b2.get(p, j);
      }
      out[i * n + j] = sum;
    }
  }

  let result = new Tensor(out, [m, n]);
  // Kembalikan ke 1D jika input 1D
  if (aIs1D && bIs1D) result = result.view(1);
  else if (aIs1D)     result = result.view(n);
  else if (bIs1D)     result = result.view(m);
  return result;
}

/** Batch matrix multiplication: [B,m,k] @ [B,k,n] → [B,m,n] */
function bmm(A: Tensor, B: Tensor): Tensor {
  if (A.ndim !== 3 || B.ndim !== 3) {
    throw new Error(`bmm: perlu tensor 3D [batch,m,k], dapat ${A.shape} dan ${B.shape}`);
  }
  const [batch, m, k] = A.shape;
  const [b2, k2, n]   = B.shape;
  if (batch !== b2 || k !== k2) {
    throw new Error(`bmm: shape tidak kompatibel: ${A.shape} @ ${B.shape}`);
  }

  const out = new Float32Array(batch * m * n);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum += A.get(b, i, p) * B.get(b, p, j);
        }
        out[b * m * n + i * n + j] = sum;
      }
    }
  }
  return new Tensor(out, [batch, m, n]);
}

// ---- Contoh Penggunaan ----------------------------------------

const A_mat = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]); // [2, 3]
const B_mat = Tensor.fromArray([[7, 8], [9, 10], [11, 12]]); // [3, 2]
const C_mat = matmul(A_mat, B_mat); // [2, 2]
console.log("matmul [2,3]@[3,2]:", C_mat.shape); // [2, 2]
console.log(C_mat.toString());
// [[58, 64],
//  [139, 154]]

// Dot product dua vektor
const u2 = new Tensor([1, 2, 3], [3]);
const v2 = new Tensor([4, 5, 6], [3]);
console.log("dot product:", matmul(u2, v2).data[0]); // 32

// Batch matmul: attention scores
const Q = new Tensor(Array.from({length: 2*3*4}, (_, i) => i * 0.1), [2, 3, 4]);
const K = new Tensor(Array.from({length: 2*4*3}, (_, i) => i * 0.1), [2, 4, 3]);
const scores = bmm(Q, K);
console.log("attention scores shape:", scores.shape); // [2, 3, 3]
```

---

## 11. `squeeze()` & `unsqueeze()` — Kelola Dimensi Tunggal

### Teori

`unsqueeze(dim)` menyisipkan dimensi baru berukuran `1` pada posisi `dim`. Berguna untuk mengubah vektor `[n]` menjadi kolom `[n, 1]` atau baris `[1, n]` agar broadcasting berjalan dengan benar.

`squeeze(dim?)` menghapus semua (atau satu) dimensi berukuran `1`. Kebalikan dari `unsqueeze`.

```
unsqueeze([3], dim=0)  → [1, 3]
unsqueeze([3], dim=1)  → [3, 1]
squeeze([1, 3, 1])     → [3]
squeeze([1, 3, 1], 0)  → [3, 1]
```

Kedua operasi ini adalah **zero-copy** — hanya memodifikasi shape dan strides.

### Implementasi

```typescript
/** Sisipkan dimensi baru berukuran 1 pada posisi dim */
function unsqueeze(t: Tensor, dim: number): Tensor {
  if (dim < 0) dim += t.ndim + 1;
  if (dim < 0 || dim > t.ndim) {
    throw new Error(`unsqueeze: dim ${dim} di luar range [0, ${t.ndim}]`);
  }
  const newShape = [...t.shape];
  newShape.splice(dim, 0, 1);               // sisipkan 1 di posisi dim
  const newStrides = [...t.strides];
  // Stride baru = stride sebelah kanan (atau 1 jika di ujung)
  const strideVal = dim < t.strides.length ? t.strides[dim] : 1;
  newStrides.splice(dim, 0, strideVal);
  const result    = new Tensor(t.data, newShape);
  result.strides  = newStrides;
  return result;
}

/** Hapus dimensi berukuran 1. Jika dim diberikan, hanya hapus dim itu */
function squeeze(t: Tensor, dim?: number): Tensor {
  let newShape:   number[];
  let newStrides: number[];

  if (dim !== undefined) {
    if (dim < 0) dim += t.ndim;
    if (t.shape[dim] !== 1) return t; // tidak ada yang di-squeeze
    newShape   = t.shape.filter((_, i) => i !== dim);
    newStrides = t.strides.filter((_, i) => i !== dim);
  } else {
    newShape   = [];
    newStrides = [];
    for (let i = 0; i < t.ndim; i++) {
      if (t.shape[i] !== 1) {
        newShape.push(t.shape[i]);
        newStrides.push(t.strides[i]);
      }
    }
    if (newShape.length === 0) { newShape = [1]; newStrides = [1]; }
  }

  const result   = new Tensor(t.data, newShape);
  result.strides = newStrides;
  return result;
}

// ---- Contoh Penggunaan ----------------------------------------

const vec = new Tensor([1, 2, 3], [3]);

const col = unsqueeze(vec, 1); // [3] → [3, 1]
console.log("unsqueeze dim=1:", col.shape, col.strides); // [3,1], [1,1]

const row = unsqueeze(vec, 0); // [3] → [1, 3]
console.log("unsqueeze dim=0:", row.shape);              // [1, 3]

const ugly = new Tensor(Array.from({length:6},(_,i)=>i), [1, 2, 1, 3]);
const clean = squeeze(ugly);
console.log("squeeze semua:", clean.shape); // [2, 3]

const partial = squeeze(ugly, 0);
console.log("squeeze dim=0:", partial.shape); // [2, 1, 3]
```

---

## 12. `cat()` & `stack()` — Gabung Tensor

### Teori

**`cat(tensors, dim)`** — Menggabungkan tensor **sepanjang dimensi yang sudah ada**. Semua dimensi kecuali `dim` harus sama.

```
cat([[1,2],[3,4]], [[5,6]], dim=0) → [[1,2],[3,4],[5,6]]  shape [3,2]
cat([[1,2],[3,4]], [[5],[6]], dim=1) → [[1,2,5],[3,4,6]]  shape [2,3]
```

**`stack(tensors, dim)`** — Menggabungkan tensor dengan **membuat dimensi baru** pada posisi `dim`. Semua tensor harus memiliki shape yang persis sama.

```
stack([[1,2],[3,4]], dim=0) → [[[1,2],[3,4]]]  shape [2,2]  (tumpuk jadi batch)
stack([[1,2],[3,4]], dim=1) → shape [2,2]      (tumpuk jadi kolom baru)
```

### Implementasi

```typescript
/** Gabungkan array tensor sepanjang dimensi dim yang sudah ada */
function cat(tensors: Tensor[], dim: number): Tensor {
  if (tensors.length === 0) throw new Error("cat: perlu setidaknya 1 tensor");

  // Normalisasi dim negatif
  const ndim = tensors[0].ndim;
  if (dim < 0) dim += ndim;

  // Validasi semua shape kecuali dim
  for (let t = 1; t < tensors.length; t++) {
    for (let d = 0; d < ndim; d++) {
      if (d !== dim && tensors[t].shape[d] !== tensors[0].shape[d]) {
        throw new Error(
          `cat: shape tidak cocok di dim ${d}: ${tensors[0].shape} vs ${tensors[t].shape}`
        );
      }
    }
  }

  // Hitung output shape
  const outShape = [...tensors[0].shape];
  outShape[dim]  = tensors.reduce((acc, t) => acc + t.shape[dim], 0);
  const outSize  = outShape.reduce((a, b) => a * b, 1);
  const outData  = new Float32Array(outSize);
  const outStrides = Tensor.computeStrides(outShape);

  // Salin data per tensor
  let offset = 0; // offset di dimensi `dim` dalam output
  for (const t of tensors) {
    const tSize = t.size;
    for (let i = 0; i < tSize; i++) {
      // Konversi flat index i ke multi-index di t
      const mi: number[] = new Array(ndim);
      let rem = i;
      for (let d = ndim - 1; d >= 0; d--) {
        mi[d] = rem % t.shape[d];
        rem   = Math.floor(rem / t.shape[d]);
      }
      // Hitung flat index di output (offset dimensi dim)
      mi[dim] += offset;
      let outFlat = 0;
      for (let d = 0; d < ndim; d++) outFlat += mi[d] * outStrides[d];
      outData[outFlat] = t.data[i];
    }
    offset += t.shape[dim];
  }

  return new Tensor(outData, outShape);
}

/** Tumpuk array tensor dengan membuat dimensi baru di posisi dim */
function stack(tensors: Tensor[], dim: number = 0): Tensor {
  if (tensors.length === 0) throw new Error("stack: perlu setidaknya 1 tensor");

  // Validasi semua shape sama
  const refShape = tensors[0].shape;
  for (const t of tensors) {
    if (t.shape.join(',') !== refShape.join(',')) {
      throw new Error(`stack: semua tensor harus punya shape sama: ${refShape} vs ${t.shape}`);
    }
  }

  // Tambahkan dimensi 1 pada tiap tensor, lalu cat
  const unsqueezed = tensors.map(t => unsqueeze(t, dim));
  return cat(unsqueezed, dim);
}

// ---- Contoh Penggunaan ----------------------------------------

const r1 = new Tensor([1, 2, 3], [1, 3]);
const r2 = new Tensor([4, 5, 6], [1, 3]);
const r3 = new Tensor([7, 8, 9], [1, 3]);

// cat sepanjang dim=0 (tambah baris)
const catted = cat([r1, r2, r3], 0);
console.log("cat dim=0 shape:", catted.shape); // [3, 3]
console.log(catted.toString());

// cat sepanjang dim=1 (tambah kolom)
const c1 = new Tensor([1, 4], [2, 1]);
const c2 = new Tensor([2, 5], [2, 1]);
const c3 = new Tensor([3, 6], [2, 1]);
const catCols = cat([c1, c2, c3], 1);
console.log("cat dim=1 shape:", catCols.shape); // [2, 3]

// stack → buat dimensi batch baru
const v1 = new Tensor([1, 2, 3], [3]);
const v2 = new Tensor([4, 5, 6], [3]);
const v3 = new Tensor([7, 8, 9], [3]);
const stacked = stack([v1, v2, v3], 0); // [3, 3] — 3 vektor jadi batch
console.log("stack dim=0 shape:", stacked.shape); // [3, 3]

const stackedCols = stack([v1, v2, v3], 1); // [3, 3] — tumpuk sebagai kolom
console.log("stack dim=1 shape:", stackedCols.shape); // [3, 3]
```

---

## 13. `clamp()` — Pembatasan Nilai

### Teori

`clamp(min, max)` memotong setiap nilai tensor ke dalam rentang `[min, max]`:

```
clamp(x, a, b) = max(a, min(b, x))
```

Jika hanya `min` diberikan: `clamp(x, a)` = `max(a, x)`.  
Jika hanya `max` diberikan: `clamp(x, max=b)` = `min(b, x)`.

**Backward pass** (gradient):
```
∂clamp/∂x = 1  jika min ≤ x ≤ max
           = 0  jika x < min atau x > max  (gradient tidak mengalir)
```

**Aplikasi umum:**
- Gradient clipping (cegah exploding gradients)
- ReLU = `clamp(x, min=0)`
- Hard Sigmoid = `clamp(x/6 + 0.5, 0, 1)`
- Batasi logit sebelum log

### Implementasi

```typescript
/** Batasi nilai tensor ke [minVal, maxVal] */
function clamp(t: Tensor, minVal?: number, maxVal?: number): Tensor {
  if (minVal !== undefined && maxVal !== undefined && minVal > maxVal) {
    throw new Error(`clamp: minVal (${minVal}) > maxVal (${maxVal})`);
  }
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    let v = t.data[i];
    if (minVal !== undefined) v = Math.max(minVal, v);
    if (maxVal !== undefined) v = Math.min(maxVal, v);
    out[i] = v;
  }
  return new Tensor(out, [...t.shape]);
}

/** ReLU = clamp(x, min=0) */
function relu(t: Tensor): Tensor {
  return clamp(t, 0);
}

/** Hard Sigmoid = clamp(x/6 + 0.5, 0, 1) */
function hardSigmoid(t: Tensor): Tensor {
  const shifted = new Float32Array(t.data.map(v => v / 6 + 0.5));
  return clamp(new Tensor(shifted, t.shape), 0, 1);
}

/** Gradient clipping by value */
function clipGradValue(t: Tensor, clipValue: number): void {
  if (t.grad) {
    t.grad = clamp(t.grad, -clipValue, clipValue);
  }
}

// ---- Contoh Penggunaan ----------------------------------------

const raw = new Tensor([-3, -1, 0, 1, 2, 5], [6]);

console.log("clamp(-1, 3):", clamp(raw, -1, 3).data);  // [-1,-1,0,1,2,3]
console.log("relu:", relu(raw).data);                   // [0,0,0,1,2,5]
console.log("hardSigmoid:", hardSigmoid(raw).data);     // [0,0.33,0.5,0.67,0.83,1]

// Gradient clipping: cegah exploding gradients
const wParam = new Tensor([0.5, 1.0, -0.3], [3]);
wParam.grad  = new Tensor([100, -200, 50], [3]); // gradient meledak
clipGradValue(wParam, 5.0);
console.log("Grad setelah clip:", wParam.grad?.data); // [5, -5, 5]
```

---

## 14. Fungsi Aktivasi — `sigmoid()`, `tanh()`, `softmax()`

### Teori

**Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))     output ∈ (0, 1)
∂σ/∂x = σ(x) * (1 - σ(x))
```

**Tanh:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))     output ∈ (-1, 1)
∂tanh/∂x = 1 - tanh(x)²
```

**Softmax** (konversi logit ke distribusi probabilitas):
```
softmax(x)_i = e^(x_i - max(x)) / Σ_j e^(x_j - max(x))
```
Pengurangan `max(x)` adalah trik numerik untuk mencegah overflow (`exp` dari angka besar).

**Log-Softmax** (lebih stabil secara numerik untuk cross-entropy loss):
```
log_softmax(x)_i = x_i - max(x) - log(Σ_j e^(x_j - max(x)))
```

### Implementasi

```typescript
/** Sigmoid: 1 / (1 + exp(-x)) */
function sigmoid(t: Tensor): Tensor {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    out[i] = 1 / (1 + Math.exp(-t.data[i]));
  }
  return new Tensor(out, [...t.shape]);
}

/** Tanh */
function tanh(t: Tensor): Tensor {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    out[i] = Math.tanh(t.data[i]);
  }
  return new Tensor(out, [...t.shape]);
}

/** Softmax sepanjang dim (numerically stable) */
function softmax(t: Tensor, dim: number = -1): Tensor {
  if (dim < 0) dim += t.ndim;

  const outShape = [...t.shape];
  const dimSize  = t.shape[dim];
  const out      = new Float32Array(t.data.length);

  // Iterasi semua "slice" sepanjang dim
  const outerSize = t.shape.slice(0, dim).reduce((a, b) => a * b, 1);
  const innerSize = t.shape.slice(dim + 1).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      // Kumpulkan nilai sepanjang dim
      const vals: number[] = [];
      for (let d = 0; d < dimSize; d++) {
        const idx = outer * dimSize * innerSize + d * innerSize + inner;
        vals.push(t.data[idx]);
      }
      // Numerically stable: kurangi max
      const maxVal = Math.max(...vals);
      const exps   = vals.map(v => Math.exp(v - maxVal));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      // Tulis kembali
      for (let d = 0; d < dimSize; d++) {
        const idx     = outer * dimSize * innerSize + d * innerSize + inner;
        out[idx] = exps[d] / sumExp;
      }
    }
  }
  return new Tensor(out, outShape);
}

/** Log-Softmax (lebih stabil untuk cross-entropy) */
function logSoftmax(t: Tensor, dim: number = -1): Tensor {
  if (dim < 0) dim += t.ndim;
  const dimSize  = t.shape[dim];
  const out      = new Float32Array(t.data.length);
  const outerSize = t.shape.slice(0, dim).reduce((a, b) => a * b, 1);
  const innerSize = t.shape.slice(dim + 1).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      const vals: number[] = [];
      for (let d = 0; d < dimSize; d++) {
        vals.push(t.data[outer * dimSize * innerSize + d * innerSize + inner]);
      }
      const maxVal   = Math.max(...vals);
      const logSumExp = Math.log(vals.reduce((s, v) => s + Math.exp(v - maxVal), 0)) + maxVal;
      for (let d = 0; d < dimSize; d++) {
        const idx    = outer * dimSize * innerSize + d * innerSize + inner;
        out[idx] = t.data[idx] - logSumExp;
      }
    }
  }
  return new Tensor(out, [...t.shape]);
}

// ---- Contoh Penggunaan ----------------------------------------

const logits2 = new Tensor([-2, -1, 0, 1, 2], [5]);

console.log("sigmoid:", sigmoid(logits2).data);
// [0.119, 0.269, 0.5, 0.731, 0.881]

console.log("tanh:", tanh(logits2).data);
// [-0.964, -0.762, 0, 0.762, 0.964]

// Softmax untuk klasifikasi
const batchLogits = new Tensor([
  1.0, 2.0, 0.5,  // sample 0
  3.0, 0.1, 1.5   // sample 1
], [2, 3]);

const probs = softmax(batchLogits, 1);
console.log("softmax shape:", probs.shape); // [2, 3]
// Setiap baris berjumlah 1:
console.log("row 0 sum:", Array.from(probs.data.slice(0,3)).reduce((a,b)=>a+b,0)); // ~1.0
console.log("row 1 sum:", Array.from(probs.data.slice(3,6)).reduce((a,b)=>a+b,0)); // ~1.0

const logProbs = logSoftmax(batchLogits, 1);
console.log("log_softmax:", logProbs.data);
```

---

## 15. `max()` & `min()` — Nilai Ekstrem & Argmax/Argmin

### Teori

`max(tensor)` → nilai maksimum global.  
`max(tensor, dim)` → nilai maksimum sepanjang dimensi tertentu, beserta indeksnya (`values`, `indices`).

`argmax(tensor, dim)` → **hanya** mengembalikan indeks nilai maksimum.

Ini digunakan untuk:
- Prediksi kelas: `argmax(logits)` → label prediksi
- Masking: `max(scores)` sebagai threshold
- Numerically stable softmax: `x - max(x)`

**Backward untuk max:**
```
∂L/∂x_i = ∂L/∂y   jika x_i == max(x)
         = 0        selainnya
```

### Implementasi

```typescript
/** Nilai maksimum global */
function globalMax(t: Tensor): number {
  let m = -Infinity;
  for (let i = 0; i < t.data.length; i++) if (t.data[i] > m) m = t.data[i];
  return m;
}

/** Nilai minimum global */
function globalMin(t: Tensor): number {
  let m = Infinity;
  for (let i = 0; i < t.data.length; i++) if (t.data[i] < m) m = t.data[i];
  return m;
}

interface ReduceResult { values: Tensor; indices: Tensor; }

/** Max/min sepanjang satu dimensi, kembalikan values dan indices */
function reduceAlongDim(
  t: Tensor,
  dim: number,
  mode: 'max' | 'min',
  keepdim: boolean = false
): ReduceResult {
  if (dim < 0) dim += t.ndim;

  const outShape = [...t.shape]; outShape[dim] = 1;
  const outSize  = outShape.reduce((a, b) => a * b, 1);
  const values   = new Float32Array(outSize).fill(mode === 'max' ? -Infinity : Infinity);
  const indices  = new Int32Array(outSize).fill(0);

  for (let i = 0; i < t.size; i++) {
    // Flat → multi-index
    const mi: number[] = new Array(t.ndim);
    let rem = i;
    for (let d = t.ndim - 1; d >= 0; d--) {
      mi[d] = rem % t.shape[d];
      rem   = Math.floor(rem / t.shape[d]);
    }
    // Hitung flat index di output
    const oi = [...mi]; oi[dim] = 0;
    let outFlat = 0, str = 1;
    for (let d = outShape.length - 1; d >= 0; d--) {
      outFlat += oi[d] * str; str *= outShape[d];
    }
    // Update
    const cond = mode === 'max'
      ? t.data[i] > values[outFlat]
      : t.data[i] < values[outFlat];
    if (cond) {
      values[outFlat]  = t.data[i];
      indices[outFlat] = mi[dim];
    }
  }

  const finalShape = keepdim ? outShape : t.shape.filter((_, d) => d !== dim);
  return {
    values:  new Tensor(values,  finalShape),
    indices: new Tensor(Array.from(indices), finalShape),
  };
}

/** argmax sepanjang dim */
function argmax(t: Tensor, dim: number, keepdim = false): Tensor {
  return reduceAlongDim(t, dim, 'max', keepdim).indices;
}

/** argmin sepanjang dim */
function argmin(t: Tensor, dim: number, keepdim = false): Tensor {
  return reduceAlongDim(t, dim, 'min', keepdim).indices;
}

// ---- Contoh Penggunaan ----------------------------------------

const scores2 = new Tensor([
  0.1, 0.7, 0.2,   // sample 0 → prediksi kelas 1
  0.8, 0.1, 0.1,   // sample 1 → prediksi kelas 0
  0.3, 0.3, 0.4    // sample 2 → prediksi kelas 2
], [3, 3]);

const preds = argmax(scores2, 1);
console.log("Prediksi kelas:", preds.data); // [1, 0, 2]

const { values: maxVals, indices: maxIdx } = reduceAlongDim(scores2, 1, 'max');
console.log("Max values:", maxVals.data);   // [0.7, 0.8, 0.4]
console.log("Max indices:", maxIdx.data);   // [1, 0, 2]

// Top confidence
console.log("Global max:", globalMax(scores2)); // 0.8
console.log("Global min:", globalMin(scores2)); // 0.1
```

---

## 16. `pow()` & `sqrt()` — Operasi Pangkat

### Teori

`pow(x, n)` = `xⁿ` element-wise.  
`sqrt(x)` = `x^0.5` element-wise (syarat: `x ≥ 0`).

**Backward:**
```
∂(xⁿ)/∂x = n · x^(n-1)
∂√x/∂x   = 1 / (2√x)
```

Digunakan di:
- Normalisasi: `x / sqrt(var + ε)`
- Attention scaling: `Q @ K.T / sqrt(d_k)`
- L2 norm: `sqrt(sum(x²))`
- RMS Norm: `x / sqrt(mean(x²) + ε)`

### Implementasi

```typescript
/** x^n element-wise */
function pow(t: Tensor, n: number): Tensor {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) out[i] = Math.pow(t.data[i], n);
  return new Tensor(out, [...t.shape]);
}

/** sqrt element-wise */
function sqrt(t: Tensor, eps: number = 0): Tensor {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    if (t.data[i] + eps < 0) throw new Error(`sqrt: nilai negatif di indeks ${i}: ${t.data[i]}`);
    out[i] = Math.sqrt(t.data[i] + eps);
  }
  return new Tensor(out, [...t.shape]);
}

/** exp element-wise */
function exp(t: Tensor): Tensor {
  const out = new Float32Array(t.data.map(Math.exp));
  return new Tensor(out, [...t.shape]);
}

/** log element-wise (natural log) */
function log(t: Tensor, eps: number = 1e-8): Tensor {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    out[i] = Math.log(Math.max(t.data[i], eps));
  }
  return new Tensor(out, [...t.shape]);
}

/** L2 norm vektor */
function norm(t: Tensor): number {
  let sum = 0;
  for (let i = 0; i < t.data.length; i++) sum += t.data[i] * t.data[i];
  return Math.sqrt(sum);
}

/** RMS Norm (digunakan di LLaMA / Mistral) */
function rmsNorm(t: Tensor, dim: number = -1, eps: number = 1e-8): Tensor {
  if (dim < 0) dim += t.ndim;
  // mean(x^2) per dim
  const xSq   = pow(t, 2);
  const meanSq = meanDim(xSq, dim, true);
  const rms    = sqrt(meanSq, eps);
  return div(t, rms);
}

// ---- Contoh Penggunaan ----------------------------------------

const vals = new Tensor([1, 4, 9, 16, 25], [5]);
console.log("pow(2):", pow(vals, 2).data);    // [1, 16, 81, 256, 625]
console.log("sqrt:", sqrt(vals).data);         // [1, 2, 3, 4, 5]

// Scaled dot-product attention
const d_k  = 64;
const scale = new Tensor([1 / Math.sqrt(d_k)], [1]);
const Q2 = Tensor.fromArray([[1.0, 0.5, -0.3, 0.8]]);
const K2 = Tensor.fromArray([[0.2, 0.9, 0.1, -0.4]]);
// scores = Q @ K.T / sqrt(d_k) -- disederhanakan untuk 1 head
const rawScore = matmul(Q2, K2.T);
const scaledScore = mul(rawScore, scale);
console.log("scaled attention score:", scaledScore.data);

// L2 norm
const embedding = new Tensor([3, 4], [2]);
console.log("L2 norm:", norm(embedding)); // 5.0

// RMS Norm (contoh sederhana)
const hiddens = new Tensor([1, 2, 3, 4], [1, 4]);
const normed  = rmsNorm(hiddens, 1);
console.log("RMS normed:", normed.data);
```

---

## 17. `flatten()` & `permute()` — Reshape Lanjut

### Teori

**`flatten(start_dim, end_dim)`** — Meratakan (merge) beberapa dimensi berurutan menjadi satu.  

```
flatten([2,3,4], 0, 1)   → [6, 4]   (gabung dim 0 dan 1)
flatten([2,3,4], 1, 2)   → [2, 12]  (gabung dim 1 dan 2)
flatten([2,3,4])          → [24]     (semua flat)
```

**`permute(order)`** — Susun ulang **semua** dimensi sekaligus, berbeda dari `transpose` yang hanya menukar dua.

```
permute([2,3,4], [2,0,1]) → [4,2,3]
permute([B,H,T,D], [0,2,1,3]) → [B,T,H,D]   # contoh di multi-head attention
```

Keduanya adalah **zero-copy** (hanya memodifikasi shape/strides).

### Implementasi

```typescript
/** Ratakan dimensi start_dim..end_dim menjadi satu */
function flatten(t: Tensor, startDim: number = 0, endDim: number = -1): Tensor {
  if (startDim < 0) startDim += t.ndim;
  if (endDim   < 0) endDim   += t.ndim;
  if (startDim > endDim) throw new Error("flatten: startDim > endDim");

  const mergedSize  = t.shape.slice(startDim, endDim + 1).reduce((a, b) => a * b, 1);
  const newShape    = [
    ...t.shape.slice(0, startDim),
    mergedSize,
    ...t.shape.slice(endDim + 1)
  ];
  return t.view(...newShape);
}

/** Susun ulang semua dimensi sesuai urutan order */
function permute(t: Tensor, order: number[]): Tensor {
  if (order.length !== t.ndim) {
    throw new Error(`permute: order harus punya ${t.ndim} elemen, dapat ${order.length}`);
  }
  // Validasi: order adalah permutasi dari 0..ndim-1
  const sorted = [...order].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i] !== i) throw new Error(`permute: order tidak valid: ${order}`);
  }

  const newShape   = order.map(d => t.shape[d]);
  const newStrides = order.map(d => t.strides[d]);

  const result    = new Tensor(t.data, newShape);
  result.strides  = newStrides;
  return result;
}

/** contiguous: salin data ke layout C-contiguous baru */
function contiguous(t: Tensor): Tensor {
  const expectedStrides = Tensor.computeStrides(t.shape);
  const isContiguous    = t.strides.every((s, i) => s === expectedStrides[i]);
  if (isContiguous) return t; // sudah contiguous, no-op

  // Salin data dengan urutan yang benar
  const out = new Float32Array(t.size);
  for (let i = 0; i < t.size; i++) {
    // multi-index dari i (pakai expected strides)
    const mi: number[] = new Array(t.ndim);
    let rem = i;
    for (let d = t.ndim - 1; d >= 0; d--) {
      mi[d] = rem % t.shape[d];
      rem   = Math.floor(rem / t.shape[d]);
    }
    // flat index di sumber (pakai strides asli)
    const srcFlat = mi.reduce((acc, idx, d) => acc + idx * t.strides[d], 0);
    out[i] = t.data[srcFlat];
  }
  return new Tensor(out, [...t.shape]);
}

// ---- Contoh Penggunaan ----------------------------------------

// flatten — konversi output conv ke vektor untuk FC layer
const convOut = new Tensor(Array.from({length:32},(_,i)=>i), [2, 4, 4]);
const flat    = flatten(convOut);
console.log("flatten total:", flat.shape); // [32]

const flatPart = flatten(convOut, 1);
console.log("flatten dim 1+:", flatPart.shape); // [2, 16]

// permute — reorder untuk multi-head attention
// Input: [batch, seq_len, n_heads, head_dim] → [batch, n_heads, seq_len, head_dim]
const mha = new Tensor(Array.from({length: 2*4*3*8}, (_,i)=>i), [2, 4, 3, 8]);
const mhaPerm = permute(mha, [0, 2, 1, 3]);
console.log("permute MHA:", mhaPerm.shape); // [2, 3, 4, 8]

// contiguous — setelah transpose/permute, paksa layout linear
const transposed = mha.transpose(1, 2);
const contig     = contiguous(transposed);
console.log("contiguous strides:", contig.strides);
// Strides seharusnya normal: [96, 24, 8, 1]
```

---

## 18. Loss Functions — `mse_loss()`, `cross_entropy_loss()`

### Teori

**Mean Squared Error (MSE):**
```
L = (1/N) Σ (ŷ_i - y_i)²

∂L/∂ŷ_i = (2/N)(ŷ_i - y_i)
```

**Cross-Entropy Loss** (untuk klasifikasi multi-kelas):
```
L = -(1/N) Σ_i Σ_c y_ic · log(p_ic)

# Untuk one-hot label (hanya satu kelas benar per sampel):
L = -(1/N) Σ_i log(p_i[true_class_i])

∂L/∂z_k = p_k - y_k    (di mana z = logit, p = softmax(z))
```

**Binary Cross-Entropy (BCE):**
```
L = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)]

∂L/∂p = -(y/p - (1-y)/(1-p)) / N
```

### Implementasi

```typescript
/** Mean Squared Error Loss */
function mseLoss(pred: Tensor, target: Tensor): Tensor {
  if (pred.size !== target.size) {
    throw new Error(`mse_loss: ukuran berbeda: ${pred.size} vs ${target.size}`);
  }
  let sum = 0;
  for (let i = 0; i < pred.data.length; i++) {
    const diff = pred.data[i] - target.data[i];
    sum += diff * diff;
  }
  return new Tensor([sum / pred.size], [1]);
}

/**
 * Cross-Entropy Loss dari logit mentah (tanpa softmax dulu).
 * @param logits  [N, C] — output model sebelum softmax
 * @param targets [N]   — label integer (indeks kelas benar)
 */
function crossEntropyLoss(logits: Tensor, targets: Tensor): Tensor {
  if (logits.ndim !== 2) throw new Error("cross_entropy: logits harus [N, C]");
  if (targets.ndim !== 1) throw new Error("cross_entropy: targets harus [N]");

  const [N, C] = logits.shape;
  const logProb = logSoftmax(logits, 1); // numerically stable
  let loss = 0;
  for (let i = 0; i < N; i++) {
    const cls = Math.round(targets.data[i]);
    if (cls < 0 || cls >= C) throw new Error(`cross_entropy: kelas ${cls} di luar [0,${C})`);
    loss -= logProb.get(i, cls);
  }
  return new Tensor([loss / N], [1]);
}

/**
 * Binary Cross-Entropy Loss.
 * @param pred    [N] — probabilitas prediksi (sudah sigmoid)
 * @param targets [N] — label biner (0 atau 1)
 */
function bceLoss(pred: Tensor, targets: Tensor, eps: number = 1e-8): Tensor {
  if (pred.size !== targets.size) {
    throw new Error("bce_loss: ukuran berbeda");
  }
  let loss = 0;
  for (let i = 0; i < pred.data.length; i++) {
    const p = Math.min(Math.max(pred.data[i], eps), 1 - eps); // clamp numerik
    const y = targets.data[i];
    loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
  }
  return new Tensor([loss / pred.size], [1]);
}

// ---- Contoh Penggunaan ----------------------------------------

// MSE
const predictions = new Tensor([2.5, 0.0, 2.1, 7.8], [4]);
const targets2    = new Tensor([3.0, -0.5, 2.0, 7.0], [4]);
console.log("MSE loss:", mseLoss(predictions, targets2).data[0].toFixed(4)); // ~0.2875

// Cross-Entropy
const classLogits = new Tensor([
  [2.0, 0.5, 0.3],  // sample 0: benar kelas 0
  [0.1, 3.0, 0.2],  // sample 1: benar kelas 1
  [0.3, 0.2, 2.5],  // sample 2: benar kelas 2
], [3, 3]);
const classLabels = new Tensor([0, 1, 2], [3]);
const ceLoss = crossEntropyLoss(
  new Tensor(classLogits.data, [3, 3]),
  classLabels
);
console.log("Cross-entropy loss:", ceLoss.data[0].toFixed(4)); // ~0.1889 (kecil, prediksi benar)

// BCE
const binPred    = new Tensor([0.9, 0.1, 0.8, 0.2], [4]);
const binTargets = new Tensor([1,   0,   1,   0  ], [4]);
console.log("BCE loss:", bceLoss(binPred, binTargets).data[0].toFixed(4)); // kecil
```

---

## 19. `where()` & `masked_fill()` — Operasi Kondisional

### Teori

**`where(condition, x, y)`** — Pilih elemen dari `x` atau `y` berdasarkan kondisi boolean:

```
out[i] = x[i]  jika condition[i] == true
       = y[i]  jika condition[i] == false
```

Ekuivalen dengan ternary operator per elemen. Sangat berguna untuk masking di attention.

**`masked_fill(mask, value)`** — Isi posisi di mana `mask == true` dengan `value` tertentu.  
Digunakan di causal attention mask untuk mengisi `-inf` sebelum softmax.

### Implementasi

```typescript
/** Tensor boolean — hanya alias, data tetap Float32 (0 = false, !=0 = true) */
type BoolTensor = Tensor;

/** where(condition, x, y): pilih dari x jika true, dari y jika false */
function where(cond: BoolTensor, x: Tensor, y: Tensor): Tensor {
  const outShape = broadcastShapes(broadcastShapes(cond.shape, x.shape), y.shape);
  const outSize  = outShape.reduce((a, b) => a * b, 1);
  const out      = new Float32Array(outSize);

  for (let i = 0; i < outSize; i++) {
    const c = getBcast(cond, outShape, i);
    out[i]  = c !== 0 ? getBcast(x, outShape, i) : getBcast(y, outShape, i);
  }
  return new Tensor(out, outShape);
}

/** masked_fill: isi posisi mask==true dengan value */
function maskedFill(t: Tensor, mask: BoolTensor, value: number): Tensor {
  if (t.size !== mask.size) {
    throw new Error("masked_fill: tensor dan mask harus berukuran sama");
  }
  const out = new Float32Array(t.data);
  for (let i = 0; i < out.length; i++) {
    if (mask.data[i] !== 0) out[i] = value;
  }
  return new Tensor(out, [...t.shape]);
}

/** Buat causal (lower-triangular) mask untuk attention */
function causalMask(seqLen: number): BoolTensor {
  const data = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      // True (1) jika j > i — posisi yang harus di-mask (future tokens)
      data[i * seqLen + j] = j > i ? 1 : 0;
    }
  }
  return new Tensor(data, [seqLen, seqLen]);
}

/** Buat tensor kondisi: elemen > threshold */
function greaterThan(t: Tensor, threshold: number): BoolTensor {
  const out = new Float32Array(t.data.map(v => v > threshold ? 1 : 0));
  return new Tensor(out, [...t.shape]);
}

/** Buat tensor kondisi: elemen < threshold */
function lessThan(t: Tensor, threshold: number): BoolTensor {
  const out = new Float32Array(t.data.map(v => v < threshold ? 1 : 0));
  return new Tensor(out, [...t.shape]);
}

// ---- Contoh Penggunaan ----------------------------------------

// where: absolute value manual
const mixedVals = new Tensor([-3, -1, 0, 2, -5, 4], [6]);
const negMask   = lessThan(mixedVals, 0);
const negated   = new Tensor(mixedVals.data.map(v => -v), mixedVals.shape);
const absVals   = where(negMask, negated, mixedVals);
console.log("abs via where:", absVals.data); // [3,1,0,2,5,4]

// Causal attention mask (4x4)
const mask4   = causalMask(4);
const attnRaw = new Tensor([
  0.1, 0.5, 0.3, 0.2,
  0.4, 0.2, 0.1, 0.3,
  0.3, 0.4, 0.2, 0.1,
  0.2, 0.3, 0.4, 0.1,
], [4, 4]);

// Isi posisi future dengan -inf sebelum softmax
const maskedAttn = maskedFill(attnRaw, mask4, -Infinity);
console.log("Masked attention (row 0):", maskedAttn.data.slice(0, 4));
// [0.1, -Inf, -Inf, -Inf]

// Softmax pada masked attention → future position = 0 setelah exp(-inf)
const attnWeights = softmax(maskedAttn, 1);
console.log("Attention weights (row 0):", attnWeights.data.slice(0, 4));
// [1.0, 0, 0, 0] — hanya bisa attend ke posisi 0
```

---

## 20. `einsum()` — Einstein Summation

### Teori

`einsum` adalah notasi universal untuk operasi tensor. Menggunakan huruf sebagai label dimensi:

```
"ij,jk->ik"   = matmul
"ii->"        = trace (jumlah diagonal)
"ij->ji"      = transpose
"ij,ij->"     = dot product semua elemen
"bi,bj->bij"  = batch outer product
"bik,bkj->bij" = batch matmul
"ij,j->i"     = matrix-vektor
"i,j->ij"     = outer product
```

Aturan:
- Indeks yang **muncul di output** → dimensi yang dipertahankan.
- Indeks yang **tidak muncul di output** → dimensi yang di-sumasi (contracted).

### Implementasi

```typescript
/**
 * Implementasi einsum minimalis.
 * Mendukung: matmul, dot, outer, transpose, trace, batch matmul,
 *            matrix-vektor, dan operasi bivariat umum.
 */
function einsum(equation: string, ...tensors: Tensor[]): Tensor {
  // Parse equation
  const [lhs, rhs] = equation.replace(/\s/g, '').split('->');
  const inputLabels = lhs.split(',');
  const outputLabel = rhs ?? '';

  if (inputLabels.length !== tensors.length) {
    throw new Error(`einsum: ${tensors.length} tensor tapi ${inputLabels.length} input di equation`);
  }

  // Validasi jumlah dimensi
  for (let t = 0; t < tensors.length; t++) {
    if (inputLabels[t].length !== tensors[t].ndim) {
      throw new Error(
        `einsum: tensor ${t} punya ndim=${tensors[t].ndim} tapi label "${inputLabels[t]}" punya ${inputLabels[t].length} karakter`
      );
    }
  }

  // Kumpulkan semua label unik
  const allLabels = Array.from(new Set(inputLabels.join('')));

  // Buat mapping label → ukuran
  const dimSize: Record<string, number> = {};
  for (let t = 0; t < tensors.length; t++) {
    for (let d = 0; d < inputLabels[t].length; d++) {
      const lbl  = inputLabels[t][d];
      const size = tensors[t].shape[d];
      if (lbl in dimSize && dimSize[lbl] !== size) {
        throw new Error(`einsum: dimensi "${lbl}" tidak konsisten: ${dimSize[lbl]} vs ${size}`);
      }
      dimSize[lbl] = size;
    }
  }

  // Tentukan output shape
  const outLabels  = outputLabel === '' ? [] : outputLabel.split('');
  const sumLabels  = allLabels.filter(l => !outLabels.includes(l));
  const outShape   = outLabels.map(l => dimSize[l]);
  const outSize    = outShape.reduce((a, b) => a * b, 1) || 1;
  const outData    = new Float32Array(outSize);

  // Iterasi semua kombinasi output + summation indices
  const allLoopLabels = [...outLabels, ...sumLabels];
  const loopSizes     = allLoopLabels.map(l => dimSize[l]);
  const totalIter     = loopSizes.reduce((a, b) => a * b, 1);

  for (let iter = 0; iter < totalIter; iter++) {
    // Dekode iter ke indeks per label
    const idxMap: Record<string, number> = {};
    let rem = iter;
    for (let d = loopSizes.length - 1; d >= 0; d--) {
      idxMap[allLoopLabels[d]] = rem % loopSizes[d];
      rem = Math.floor(rem / loopSizes[d]);
    }

    // Hitung produk dari semua input tensor
    let prod = 1;
    for (let t = 0; t < tensors.length; t++) {
      const tIndices = inputLabels[t].split('').map(l => idxMap[l]);
      prod *= tensors[t].get(...tIndices);
    }

    // Hitung output flat index
    let outFlat = 0;
    if (outLabels.length > 0) {
      let str = 1;
      for (let d = outLabels.length - 1; d >= 0; d--) {
        outFlat += idxMap[outLabels[d]] * str;
        str     *= dimSize[outLabels[d]];
      }
    }
    outData[outFlat] += prod;
  }

  return new Tensor(outData, outShape.length > 0 ? outShape : [1]);
}

// ---- Contoh Penggunaan ----------------------------------------

const E = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);       // [2, 3]
const F = Tensor.fromArray([[1, 2], [3, 4], [5, 6]]);     // [3, 2]

// matmul
const G = einsum("ij,jk->ik", E, F);
console.log("einsum matmul:", G.shape, G.data); // [2,2]: [22,28,49,64]

// transpose
const ET = einsum("ij->ji", E);
console.log("einsum transpose:", ET.shape); // [3, 2]

// dot product (sum semua elemen perkalian)
const a3 = new Tensor([1, 2, 3], [3]);
const b3 = new Tensor([4, 5, 6], [3]);
console.log("einsum dot:", einsum("i,i->", a3, b3).data[0]); // 32

// outer product
const outer_e = einsum("i,j->ij", a3, b3);
console.log("einsum outer:", outer_e.shape); // [3, 3]

// Scaled dot-product attention score (batch, heads, seq, seq)
const q = new Tensor(Array.from({length: 2*2*3*4}, (_, i) => i * 0.01), [2, 2, 3, 4]);
const k = new Tensor(Array.from({length: 2*2*3*4}, (_, i) => i * 0.01), [2, 2, 3, 4]);
const attnScores = einsum("bhid,bhjd->bhij", q, k); // [2,2,3,3]
console.log("attention einsum:", attnScores.shape);  // [2, 2, 3, 3]
```

---

## Ringkasan Semua Operasi

| No | Fungsi | PyTorch Equiv | Kegunaan Utama |
|----|--------|--------------|----------------|
| 1 | `Tensor` | `torch.Tensor` | Struktur data dasar |
| 2 | `.view()` | `.view()` / `.reshape()` | Ubah shape tanpa copy |
| 3 | `.transpose()` | `.transpose()` / `.T` | Tukar dua dimensi |
| 4 | `.backward()` | `.backward()` | Autograd / backprop |
| 4b | `zeroGrad()` / `zeroGradParams()` | `optimizer.zero_grad()` | Reset gradien seluruh graf sebelum iterasi baru |
| 5 | `add()` | `torch.add` | Penjumlahan + broadcast |
| 6 | `mul()` | `torch.mul` | Perkalian element-wise |
| 7 | `sub()` | `torch.sub` | Pengurangan + broadcast |
| 8 | `div()` | `torch.div` | Pembagian + broadcast |
| 9 | `outer()` | `torch.outer` | Produk luar dua vektor |
| 10 | `matmul()` / `bmm()` | `torch.matmul` / `torch.bmm` | Perkalian matriks |
| 11 | `squeeze()` / `unsqueeze()` | `torch.squeeze` | Kelola dimensi 1 |
| 12 | `cat()` / `stack()` | `torch.cat` / `torch.stack` | Gabungkan tensor |
| 13 | `clamp()` | `torch.clamp` | Batasi nilai, gradient clip |
| 14 | `sigmoid()` / `tanh()` / `softmax()` | `torch.sigmoid` / `F.softmax` | Aktivasi |
| 15 | `argmax()` / `argmin()` | `torch.argmax` | Prediksi kelas, indeks ekstrem |
| 16 | `pow()` / `sqrt()` / `exp()` / `log()` | `torch.pow` dll | Operasi matematika |
| 17 | `flatten()` / `permute()` | `torch.flatten` / `.permute()` | Reshape lanjut |
| 18 | `mseLoss()` / `crossEntropyLoss()` | `F.mse_loss` / `F.cross_entropy` | Fungsi loss |
| 19 | `where()` / `maskedFill()` | `torch.where` / `.masked_fill_` | Operasi kondisional / masking |
| 20 | `einsum()` | `torch.einsum` | Notasi universal tensor |

---

## Referensi & Bacaan Lanjut

| Topik | Sumber |
|-------|--------|
| PyTorch Tensor internals | [pytorch.org/docs](https://pytorch.org/docs/stable/tensors.html) |
| Autograd from scratch | Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd) |
| Broadcasting rules | [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| Strides & memory layout | [NumPy ndarray internals](https://numpy.org/doc/stable/reference/arrays.ndarray.html) |
| Backprop math | [CS231n Notes](https://cs231n.github.io/optimization-2/) |
| Einstein summation | [NumPy einsum docs](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) |
| Attention mechanism | [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762) |
| RMS Normalization | [Root Mean Square Layer Norm (Zhang et al.)](https://arxiv.org/abs/1910.07467) |

---

*Dibuat dengan TypeScript murni, tanpa dependency eksternal.*
