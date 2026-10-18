// port-lint: source nlp/tinystories.rs
package io.github.kotlinmania.candledatasets.nlp.tinystories

/**
 * Helper functions for the tinystories dataset. This uses the pre-tokenized version as generated
 * by the tools from https://github.com/karpathy/llama2.c
 */

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.readByteArray
import io.github.kotlinmania.rand.rngs.rng
import io.github.kotlinmania.rand.seq.shuffle

public class Dataset(
    public val validTokens: List<ByteArray>,
    public val trainTokens: List<ByteArray>,
) {
    public fun trainTokens(): Int = trainTokens.size

    public fun validTokens(): Int = validTokens.size

    public companion object {
        public fun mmapFile(p: Path): ByteArray {
            val source = SystemFileSystem.source(p).buffered()
            return source.readByteArray()
        }

        public fun new(dir: Path): Dataset {
            val binFiles = ArrayList<Path>()
            for (file in SystemFileSystem.list(dir)) {
                if (file.name.endsWith(".bin")) {
                    binFiles.add(file)
                }
            }
            if (binFiles.size < 2) {
                bail("found less than two bin files in $dir")
            }
            binFiles.sortBy { it.name }
            val validTokens = mmapFile(binFiles[0])
            val trainTokens = binFiles.subList(1, binFiles.size).map { mmapFile(it) }
            return Dataset(
                validTokens = listOf(validTokens),
                trainTokens = trainTokens,
            )
        }

        public fun new(dir: String): Dataset = new(Path(dir))
    }
}

public class DatasetRandomIter(
    private val allTokens: List<ByteArray>,
    private var tokens: MutableList<ByteArray>,
    private var currentTokens: ByteArray,
    private var indexesInBytes: MutableList<Int>,
    private val seqLen: Int,
    private val device: Device,
) : Iterator<CandleResult<Pair<Tensor, Tensor>>?> {

    public companion object {
        public fun new(ds: Dataset, valid: Boolean, seqLen: Int, device: Device): DatasetRandomIter {
            val allTokens = if (valid) ds.validTokens else ds.trainTokens
            val tokens = allTokens.toMutableList()
            tokens.shuffle(rng())
            val currentTokens = tokens.removeAt(tokens.size - 1)
            val seqLenInBytes = seqLen * 2
            val indexesInBytes = (0 until currentTokens.size - seqLenInBytes step seqLenInBytes).toMutableList()
            indexesInBytes.shuffle(rng())
            return DatasetRandomIter(
                allTokens = allTokens,
                tokens = tokens,
                currentTokens = currentTokens,
                indexesInBytes = indexesInBytes,
                seqLen = seqLen,
                device = device,
            )
        }
    }

    override fun hasNext(): Boolean = true

    override fun next(): CandleResult<Pair<Tensor, Tensor>>? {
        val seqLen = this.seqLen
        if (indexesInBytes.isEmpty()) {
            if (tokens.isEmpty()) {
                tokens = allTokens.toMutableList()
                tokens.shuffle(rng())
            }
            if (tokens.isEmpty()) {
                return null
            }
            currentTokens = tokens.removeAt(tokens.size - 1)
            val seqLenInBytes = seqLen * 2
            indexesInBytes = (0 until currentTokens.size - seqLenInBytes step seqLenInBytes).toMutableList()
            indexesInBytes.shuffle(rng())
        }
        if (indexesInBytes.isEmpty()) {
            return null
        }
        val startIdx = indexesInBytes.removeAt(indexesInBytes.size - 1)
        val endIdx = startIdx + 2 * (seqLen + 1)
        if (endIdx > currentTokens.size) {
            return null
        }
        val tokenCount = seqLen + 1
        val decodedTokens = UIntArray(tokenCount)
        for (i in 0 until tokenCount) {
            val offset = startIdx + i * 2
            val u16 = LittleEndian.readU16(currentTokens, offset).toUInt()
            decodedTokens[i] = u16
        }
        val inputTokens = decodedTokens.copyOfRange(0, seqLen)
        val targetTokens = decodedTokens.copyOfRange(1, tokenCount)
        return try {
            val inputs = Tensor.new(inputTokens, Shape(seqLen), device)
            val targets = Tensor.new(targetTokens, Shape(seqLen), device)
            CandleResult.Ok(Pair(inputs, targets))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }
}
