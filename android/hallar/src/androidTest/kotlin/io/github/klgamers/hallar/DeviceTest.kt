package io.github.klgamers.hallar

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class DeviceTest {
    private fun fixture(): File {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val out = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "tiny.hallar")
        ctx.assets.open("tiny.hallar").use { src -> out.outputStream().use { src.copyTo(it) } }
        return out
    }

    @Test
    fun searchOnDevice() {
        HallarModel.open(fixture()).use { m ->
            HallarIndex(m).use { idx ->
                idx.add("0", "el perro y el gato")
                idx.add("1", "receta de croquetas de jamon")
                idx.add("2", "café 🚀 boleto de tren")
                assertEquals("1", idx.search("croquetas jamón", 1).first().id)
                assertEquals("café 🚀 boleto de tren", idx.search("tren", 1, SearchMode.KEYWORD).first().text)
                val dir = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "idx")
                idx.save(dir)
                HallarIndex.load(m, dir).use { back -> assertEquals(3, back.count) }
                assertTrue(m.embed("hola").size == 8)
            }
        }
    }
}
