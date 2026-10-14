package io.github.klgamers.hallar

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs

class HallarTest {
    private val fixture = File(System.getProperty("hallar.fixture"))
    private val docs = listOf(
        "0" to "el perro y el gato",
        "1" to "cafe con leche caliente",
        "2" to "receta de croquetas de jamon",
        "3" to "boleto de tren para el viaje",
    )

    private fun index(model: HallarModel) = HallarIndex(model).apply { addAll(docs) }

    @Test
    fun model() {
        HallarModel.open(fixture).use { m ->
            assertEquals(8, m.dimension)
            assertEquals(32, m.vocabularySize)
            assertArrayEquals(intArrayOf(2, 3, 4, 23), m.tokenize("Hola mundos!"))
            val v = m.embed("hola mundo")
            assertTrue(abs(v.sumOf { (it * it).toDouble() } - 1.0) < 1e-5)
            assertTrue(m.embed("zzz").all { it == 0f })
        }
    }

    @Test
    fun errors() {
        val e = assertThrows(HallarException::class.java) { HallarModel.open(File("/nope/x.hallar")) }
        assertTrue(e.message!!.contains("cannot open"))
        HallarModel.open(fixture).use { m ->
            index(m).use { idx ->
                assertThrows(HallarException::class.java) { idx.add("0", "dup") }
                assertThrows(HallarException::class.java) { idx.setVectorWeight(2f) }
            }
        }
    }

    @Test
    fun searchModes() {
        HallarModel.open(fixture).use { m ->
            index(m).use { idx ->
                assertEquals(4, idx.count)
                assertTrue("2" in idx)
                assertFalse("9" in idx)
                assertEquals("2", idx.search("croquetas de jamón", 1, SearchMode.KEYWORD).first().id)
                assertEquals("3", idx.search("boleto tren").first().id)
                assertEquals(2, idx.search("perro", 2, SearchMode.VECTOR).size)
                assertTrue(idx.search("zzz", mode = SearchMode.KEYWORD).isEmpty())
            }
        }
    }

    @Test
    fun unicodeRoundTrip() {
        HallarModel.open(fixture).use { m ->
            HallarIndex(m).use { idx ->
                val text = "café 🚀 con leche 👨‍👩‍👧 ñandú 数学"
                idx.add("emoji-🔥", text)
                val hit = idx.search("cafe leche", 1, SearchMode.KEYWORD).single()
                assertEquals("emoji-🔥", hit.id)
                assertEquals(text, hit.text)
                assertTrue("emoji-🔥" in idx)
            }
        }
    }

    @Test
    fun saveLoad() {
        HallarModel.open(fixture).use { m ->
            val dir = Files.createTempDirectory("hallar-kt").toFile()
            try {
                index(m).use { idx ->
                    idx.save(dir)
                    HallarIndex.load(m, dir).use { back ->
                        assertEquals(4, back.count)
                        for (q in listOf("perro", "cafe", "viaje")) assertEquals(idx.search(q), back.search(q))
                    }
                }
            } finally {
                dir.deleteRecursively()
            }
        }
    }

    @Test
    fun closedHandles() {
        val m = HallarModel.open(fixture)
        val idx = index(m)
        idx.close()
        idx.close()
        assertThrows(IllegalStateException::class.java) { idx.search("x") }
        m.close()
        assertThrows(IllegalStateException::class.java) { m.dimension }
    }

    @Test
    fun concurrentReads() {
        HallarModel.open(fixture).use { m ->
            index(m).use { idx ->
                val want = idx.search("tren")
                val pool = Executors.newFixedThreadPool(8)
                val futures = (0 until 64).map { pool.submit<List<SearchResult>> { idx.search("tren") } }
                futures.forEach { assertEquals(want, it.get()) }
                pool.shutdown()
                assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS))
            }
        }
    }
}
