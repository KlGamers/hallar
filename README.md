# hallar

[![ci](https://github.com/KlGamers/hallar/actions/workflows/ci.yml/badge.svg)](https://github.com/KlGamers/hallar/actions/workflows/ci.yml)

Búsqueda semántica e híbrida que funciona **sin internet**, dentro de apps de
iOS, Android y Flutter. Convierte texto en vectores en el propio teléfono,
combina esa búsqueda con palabras clave y guarda todo en un índice local.

```swift
let model = try HallarModel(url: modelURL)
let index = try HallarIndex(model: model)
try index.add(id: "42", text: "¿Cómo cambio la contraseña de mi cuenta?")
let hits = try index.search("olvidé mi clave")
```

## El problema

Casi todas las soluciones de búsqueda semántica suponen un servidor: mandas el
texto a una API de embeddings, guardas los vectores en una base en la nube y
consultas por red. Eso deja fuera a las apps que necesitan funcionar en el
metro, en zonas rurales, con datos caros o con información privada que no debe
salir del teléfono. Las alternativas locales suelen exigir un runtime de ML y
modelos de cientos de megabytes, y casi ninguna está pensada para español.

hallar corre completo en el dispositivo, en C++ sin dependencias, con modelos
de embeddings estáticos de 26 MB que no necesitan GPU ni runtime de ML.

## Qué incluye

| Pieza | Qué hace |
|---|---|
| Núcleo en C++17 | Tokenizadores WordPiece y Unigram de SentencePiece, embeddings, índice HNSW, BM25 y búsqueda híbrida |
| SDK de Swift | Swift Package para iOS 15 y macOS 12 |
| SDK de Android | Librería en Kotlin con JNI para arm64, armv7 y x86_64 |
| Plugin de Flutter | Bindings con `dart:ffi` para Android e iOS |
| `hallar-convert` | Convierte modelos `model2vec`, recorta su vocabulario y los cuantiza a 8 bits |

El índice vectorial es [vecdex](https://github.com/KlGamers/vecdex), incluido
en `core/third_party`.

## Resultados

### Idéntico al original

Los embeddings se calculan exactamente igual que la librería de referencia
`model2vec` de Python, token por token:

| Modelo | Tokenizador | Textos comparados | Tokens idénticos | Coseno mínimo |
|---|---|---|---|---|
| potion-base-8M | WordPiece | 20,015 | 100% | 1.000000 |
| potion-multilingual-128M | Unigram | 41,458 | 100% | 1.000000 |

Los textos incluyen Wikipedia en español, preguntas de XQuAD, CLINC150 y casos
difíciles: acentos combinados, ancho completo, ligaduras, emojis compuestos,
banderas, coreano, chino, griego, cirílico, hindi y caracteres invisibles.

### Búsqueda en español

Tres conjuntos públicos:

- **XQuAD-es:** preguntas escritas leyendo el párrafo.
- **MIRACL-es:** preguntas naturales sobre 146,750 pasajes de Wikipedia.
- **XPQA-es:** preguntas reales de compradores sobre productos.

Las preguntas de cada conjunto se dividen en dos mitades fijas. El peso de la
búsqueda híbrida se eligió con la primera mitad y los números son de la
segunda. Métrica: nDCG@10 y recall@10. Modelo: `potion-multilingual-128M`
recortado a español y en 8 bits (26 MB).

| conjunto | documentos | preguntas | BM25 | vectores | híbrida | BM25 R@10 | vectores R@10 | híbrida R@10 |
|---|---|---|---|---|---|---|---|---|
| XQuAD-es | 240 | 586 | 0.950 | 0.831 | 0.947 | 0.983 | 0.939 | 0.988 |
| MIRACL-es | 146,750 | 321 | 0.252 | 0.136 | 0.258 | 0.291 | 0.184 | 0.310 |
| XPQA-es | 1,941 | 402 | 0.463 | 0.442 | 0.484 | 0.512 | 0.503 | 0.533 |

Qué dicen los números:

- **La búsqueda híbrida gana donde la gente usa sus propias palabras.** En
  XPQA mejora 4.5% sobre BM25 y en MIRACL 2.4%, con más recall en ambos.
- **BM25 es muy difícil de vencer cuando la pregunta repite el texto.** En
  XQuAD los embeddings solos pierden contra palabras clave y la híbrida
  empata. Por eso hallar no ofrece solo búsqueda vectorial.
- **Mediana por consulta en una laptop Apple Silicon:** 0.08 ms con BM25 y
  0.14 ms con búsqueda vectorial o híbrida, incluido el cálculo del embedding.
- **Construir el índice de 146,750 pasajes** tarda 147 s en un solo hilo.

### Comprimir sin perder calidad

| Modelo | Tamaño | XQuAD-es vectores | XPQA-es vectores |
|---|---|---|---|
| multilingüe completo, 32 bits | 524.5 MB | 0.831 | 0.452 |
| recortado a español, 32 bits | 96.8 MB | 0.831 | 0.442 |
| recortado a español, 8 bits | 26.1 MB | 0.831 | 0.442 |

El recorte conserva los 92,613 tokens que aparecen en 300,000 párrafos de
Wikipedia en español. Cuantizar a 8 bits no cuesta nada medible. El recorte
cuesta 2% en preguntas sobre productos, cuyo vocabulario no siempre está en
Wikipedia; puedes recortar con textos de tu propio dominio.

## Instalación

### iOS y macOS

En Xcode, *File → Add Package Dependencies* con
`https://github.com/KlGamers/hallar`, o en `Package.swift`:

```swift
.package(url: "https://github.com/KlGamers/hallar", branch: "main")
```

### Android

```kotlin
val model = HallarModel.open(File(filesDir, "potion-es-i8.hallar"))
val index = HallarIndex(model)
index.add("42", "¿Cómo cambio la contraseña de mi cuenta?")
val hits = index.search("olvidé mi clave", limit = 5)
```

Agrega el módulo `android/hallar` a tu proyecto. Requiere minSdk 24.

### Flutter

```yaml
dependencies:
  hallar:
    git:
      url: https://github.com/KlGamers/hallar
      path: flutter
```

```dart
final index = HallarIndex(HallarModel.open(path))..add('42', '¿Cómo cambio mi contraseña?');
final hits = index.search('olvidé mi clave', limit: 5);
```

## Preparar un modelo

```sh
pip install ./tools
hallar-convert minishlab/potion-multilingual-128M potion-es-i8.hallar --quant i8 --keep-from textos.txt
```

`--keep-from` recibe archivos de texto de tu dominio y conserva solo el
vocabulario que usan. Sin él, el modelo multilingüe pesa 524 MB.

Modelos probados:

| Modelo | Idiomas | Tamaño en 8 bits |
|---|---|---|
| `minishlab/potion-base-8M` | inglés | 8.0 MB |
| `minishlab/potion-multilingual-128M`, recortado a español | español | 26.1 MB |

Funciona con cualquier modelo `model2vec` cuyo tokenizador sea WordPiece con
normalizador de BERT, o Unigram con normalizador precompilado de
SentencePiece y pre-tokenizador Metaspace. El convertidor rechaza con un
mensaje claro lo que no soporta.

## API

| Swift | Kotlin | Dart | Qué hace |
|---|---|---|---|
| `HallarModel(url:)` | `HallarModel.open(file)` | `HallarModel.open(path)` | Carga un modelo `.hallar` |
| `model.embed(_:)` | `model.embed(text)` | `model.embed(text)` | Vector normalizado de un texto |
| `HallarIndex(model:)` | `HallarIndex(model)` | `HallarIndex(model)` | Índice vacío |
| `index.add(id:text:)` | `index.add(id, text)` | `index.add(id, text)` | Agrega un documento |
| `index.search(_:limit:mode:)` | `index.search(q, limit, mode)` | `index.search(q, limit:, mode:)` | Busca en modo `hybrid`, `vector` o `keyword` |
| `index.save(to:)` | `index.save(dir)` | `index.save(dir)` | Guarda el índice en una carpeta |
| `HallarIndex.load(model:from:)` | `HallarIndex.load(model, dir)` | `HallarIndex.load(model, dir)` | Lo vuelve a abrir |
| `index.setVectorWeight(_:)` | `index.setVectorWeight(w)` | `index.setVectorWeight(w)` | Peso de los vectores en la búsqueda híbrida |

Las búsquedas se pueden hacer desde varios hilos a la vez. Agregar documentos
toma un candado exclusivo. Un índice guardado recuerda con qué modelo se
construyó y se niega a cargarse con otro.

## Cómo funciona

```
texto ─> normalización ─> tokenizador ─> promedio de vectores ─┐
                                                               ├─> HNSW (vecdex) ──┐
                         términos sin acentos ─> BM25 ─────────┘                    ├─> fusión por rango ─> resultados
                                                                                    │
```

- **Tokenizadores exactos.** WordPiece con el normalizador de BERT, y Unigram
  con el mapa de caracteres precompilado de SentencePiece, segmentación en
  grafemas de Unicode y Viterbi con puntajes en doble precisión, igual que la
  implementación de referencia.
- **Tablas Unicode generadas** a partir de la base de datos oficial de Unicode
  16, sin depender de ICU.
- **Búsqueda híbrida** con Reciprocal Rank Fusion ponderada entre BM25 y el
  índice vectorial. El peso por defecto (0.15) salió de la mitad de
  desarrollo de los tres conjuntos.
- **Formato de archivo propio** con validación de cada sección al cargar.

## Pruebas

| Capa | Qué se prueba |
|---|---|
| Núcleo C++ | Normalización, ambos tokenizadores, embeddings, cuantización, archivo corrupto, búsqueda híbrida, concurrencia y API en C, con AddressSanitizer y UBSan |
| Paridad | 61,473 textos contra `model2vec` en dos modelos |
| Swift | `swift test` en macOS; compila para simulador y dispositivo iOS |
| Kotlin | Pruebas en la JVM con la librería nativa real y prueba instrumentada en Android 15 (arm64) |
| Flutter | Pruebas con la librería nativa y prueba de integración en el emulador de Android y el simulador de iOS |
| Convertidor | Traducción de ambos formatos de tokenizador, recorte y archivos generados |

## Límites

- No se pueden borrar documentos de un índice; hay que reconstruirlo.
- La construcción del índice usa un solo hilo.
- En Flutter para iOS el plugin usa CocoaPods; falta el soporte de Swift
  Package Manager que Flutter pedirá en el futuro.
- En ARM de 32 bits las distancias usan la versión escalar, sin NEON.
- Los modelos se convierten una vez en una computadora; la conversión no corre
  en el teléfono.

## Ecosistema

hallar es parte de un conjunto de herramientas para software local-first:

| Repo | Papel |
|---|---|
| [vecdex](https://github.com/KlGamers/vecdex) | Índice vectorial HNSW que usa hallar |
| [rumbo](https://github.com/KlGamers/rumbo) | Router de intenciones con embeddings y evaluación |
| [guardia](https://github.com/KlGamers/guardia) | Revisión antes de cada commit |
| [pulso](https://github.com/KlGamers/pulso) | Monitoreo de servicios autohospedados |

## Licencia

MIT. Los modelos `potion` son de MinishLab (MIT). Los conjuntos de datos de
los benchmarks se descargan aparte y conservan sus licencias: XQuAD (CC BY-SA
4.0) y MIRACL (Apache 2.0); para XPQA consulta la licencia en su repositorio original.
