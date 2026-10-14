package io.github.klgamers.hallar

import java.io.File

class HallarException(message: String) : RuntimeException(message)

enum class SearchMode(internal val code: Int) { HYBRID(0), VECTOR(1), KEYWORD(2) }

data class SearchResult(val id: String, val text: String, val score: Float)

internal object Native {
    init {
        System.loadLibrary("hallar_jni")
    }

    external fun modelOpen(path: String): Long
    external fun modelFree(ptr: Long)
    external fun modelDim(ptr: Long): Int
    external fun modelVocab(ptr: Long): Int
    external fun embed(ptr: Long, text: String): FloatArray
    external fun tokenize(ptr: Long, text: String): IntArray
    external fun engineNew(model: Long, connections: Int, effort: Int): Long
    external fun engineLoad(model: Long, dir: String): Long
    external fun engineFree(ptr: Long)
    external fun engineAdd(ptr: Long, id: String, text: String)
    external fun engineSearch(ptr: Long, query: String, limit: Int, effort: Int, mode: Int): Array<SearchResult>
    external fun engineSave(ptr: Long, dir: String)
    external fun engineSize(ptr: Long): Int
    external fun engineHas(ptr: Long, id: String): Boolean
    external fun engineSetWeight(ptr: Long, weight: Float)
}

class HallarModel private constructor(ptr: Long) : AutoCloseable {
    private var raw = ptr

    internal val ptr: Long
        get() = raw.takeIf { it != 0L } ?: throw IllegalStateException("model is closed")

    val dimension: Int get() = Native.modelDim(ptr)
    val vocabularySize: Int get() = Native.modelVocab(ptr)

    fun embed(text: String): FloatArray = Native.embed(ptr, text)

    fun tokenize(text: String): IntArray = Native.tokenize(ptr, text)

    @Synchronized
    override fun close() {
        if (raw != 0L) Native.modelFree(raw)
        raw = 0L
    }

    companion object {
        fun open(file: File): HallarModel = HallarModel(Native.modelOpen(file.absolutePath))
    }
}

class HallarIndex private constructor(val model: HallarModel, ptr: Long) : AutoCloseable {
    private var raw = ptr

    private val ptr: Long
        get() = raw.takeIf { it != 0L } ?: throw IllegalStateException("index is closed")

    constructor(model: HallarModel, connections: Int = 16, buildEffort: Int = 200) :
        this(model, Native.engineNew(model.ptr, connections, buildEffort))

    val count: Int get() = Native.engineSize(ptr)

    operator fun contains(id: String): Boolean = Native.engineHas(ptr, id)

    fun add(id: String, text: String) = Native.engineAdd(ptr, id, text)

    fun addAll(docs: Iterable<Pair<String, String>>) = docs.forEach { (id, text) -> add(id, text) }

    fun setVectorWeight(weight: Float) = Native.engineSetWeight(ptr, weight)

    fun search(query: String, limit: Int = 10, mode: SearchMode = SearchMode.HYBRID, effort: Int = 64): List<SearchResult> =
        Native.engineSearch(ptr, query, limit, effort, mode.code).toList()

    fun save(dir: File) = Native.engineSave(ptr, dir.absolutePath)

    @Synchronized
    override fun close() {
        if (raw != 0L) Native.engineFree(raw)
        raw = 0L
    }

    companion object {
        fun load(model: HallarModel, dir: File): HallarIndex = HallarIndex(model, Native.engineLoad(model.ptr, dir.absolutePath))
    }
}
