// port-lint: source batcher.rs
package io.github.kotlinmania.candledatasets

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.Tensor

public open class Batcher<I>(
    public val inner: I,
    public var batchSize: Int = 16,
    public var returnLastIncompleteBatch: Boolean = false,
) {
    public open fun batchSize(batchSize: Int): Batcher<I> {
        this.batchSize = batchSize
        return this
    }

    public open fun returnLastIncompleteBatch(r: Boolean): Batcher<I> {
        this.returnLastIncompleteBatch = r
        return this
    }

    public companion object {
        public fun <I> new(inner: I): Batcher<I> = Batcher(inner)

        public fun new1(inner: Iterator<Tensor>): Batcher1 =
            Batcher1(Iter1(inner))

        public fun new2(inner: Iterator<Pair<Tensor, Tensor>>): Batcher2 =
            Batcher2(Iter2(inner))

        public fun newR1(inner: Iterator<CandleResult<Tensor>>): BatcherResult1 =
            BatcherResult1(IterResult1(inner))

        public fun newR2(inner: Iterator<CandleResult<Pair<Tensor, Tensor>>>): BatcherResult2 =
            BatcherResult2(IterResult2(inner))
    }
}

public class Iter1 internal constructor(
    public val inner: Iterator<Tensor>,
)

public class Iter2 internal constructor(
    public val inner: Iterator<Pair<Tensor, Tensor>>,
)

public class IterResult1 internal constructor(
    public val inner: Iterator<CandleResult<Tensor>>,
)

public class IterResult2 internal constructor(
    public val inner: Iterator<CandleResult<Pair<Tensor, Tensor>>>,
)

public class Batcher1 internal constructor(
    inner: Iter1,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter1>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>> {
    private var nextBatch: CandleResult<Tensor>? = null
    private var done: Boolean = false

    override fun batchSize(batchSize: Int): Batcher1 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): Batcher1 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    private fun prepareNext() {
        if (nextBatch != null || done) return
        val items = ArrayList<Tensor>(batchSize)
        while (items.size < batchSize) {
            if (inner.inner.hasNext()) {
                items.add(inner.inner.next())
            } else {
                if (returnLastIncompleteBatch && items.isNotEmpty()) {
                    break
                } else {
                    done = true
                    return
                }
            }
        }
        if (items.isEmpty()) {
            done = true
        } else {
            nextBatch =
                try {
                    CandleResult.Ok(Tensor.stack(items, 0))
                } catch (e: CandleException) {
                    CandleResult.Err(e)
                }
        }
    }

    override fun hasNext(): Boolean {
        prepareNext()
        return nextBatch != null
    }

    override fun next(): CandleResult<Tensor> {
        prepareNext()
        val batch = nextBatch ?: throw NoSuchElementException("No more elements in Batcher")
        nextBatch = null
        return batch
    }
}

public class Batcher2 internal constructor(
    inner: Iter2,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter2>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>> {
    private var nextBatch: CandleResult<Pair<Tensor, Tensor>>? = null
    private var done: Boolean = false

    override fun batchSize(batchSize: Int): Batcher2 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): Batcher2 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    private fun prepareNext() {
        if (nextBatch != null || done) return
        val xs = ArrayList<Tensor>(batchSize)
        val ys = ArrayList<Tensor>(batchSize)
        while (xs.size < batchSize) {
            if (inner.inner.hasNext()) {
                val pair = inner.inner.next()
                xs.add(pair.first)
                ys.add(pair.second)
            } else {
                if (returnLastIncompleteBatch && xs.isNotEmpty() && ys.isNotEmpty()) {
                    break
                } else {
                    done = true
                    return
                }
            }
        }
        if (xs.isEmpty() || ys.isEmpty()) {
            done = true
        } else {
            nextBatch =
                try {
                    val xsStacked = Tensor.stack(xs, 0)
                    val ysStacked = Tensor.stack(ys, 0)
                    CandleResult.Ok(Pair(xsStacked, ysStacked))
                } catch (e: CandleException) {
                    CandleResult.Err(e)
                }
        }
    }

    override fun hasNext(): Boolean {
        prepareNext()
        return nextBatch != null
    }

    override fun next(): CandleResult<Pair<Tensor, Tensor>> {
        prepareNext()
        val batch = nextBatch ?: throw NoSuchElementException("No more elements in Batcher")
        nextBatch = null
        return batch
    }
}

public class BatcherResult1 internal constructor(
    inner: IterResult1,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult1>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>> {
    private var nextBatch: CandleResult<Tensor>? = null
    private var done: Boolean = false

    override fun batchSize(batchSize: Int): BatcherResult1 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): BatcherResult1 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    private fun prepareNext() {
        if (nextBatch != null || done) return
        val items = ArrayList<Tensor>(batchSize)
        while (items.size < batchSize) {
            if (inner.inner.hasNext()) {
                when (val item = inner.inner.next()) {
                    is CandleResult.Ok -> {
                        items.add(item.value)
                    }

                    is CandleResult.Err -> {
                        nextBatch = item
                        return
                    }
                }
            } else {
                if (returnLastIncompleteBatch && items.isNotEmpty()) {
                    break
                } else {
                    done = true
                    return
                }
            }
        }
        if (items.isEmpty()) {
            done = true
        } else {
            nextBatch =
                try {
                    CandleResult.Ok(Tensor.stack(items, 0))
                } catch (e: CandleException) {
                    CandleResult.Err(e)
                }
        }
    }

    override fun hasNext(): Boolean {
        prepareNext()
        return nextBatch != null
    }

    override fun next(): CandleResult<Tensor> {
        prepareNext()
        val batch = nextBatch ?: throw NoSuchElementException("No more elements in Batcher")
        nextBatch = null
        return batch
    }
}

public class BatcherResult2 internal constructor(
    inner: IterResult2,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult2>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>> {
    private var nextBatch: CandleResult<Pair<Tensor, Tensor>>? = null
    private var done: Boolean = false

    override fun batchSize(batchSize: Int): BatcherResult2 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): BatcherResult2 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    private fun prepareNext() {
        if (nextBatch != null || done) return
        val xs = ArrayList<Tensor>(batchSize)
        val ys = ArrayList<Tensor>(batchSize)
        while (xs.size < batchSize) {
            if (inner.inner.hasNext()) {
                when (val item = inner.inner.next()) {
                    is CandleResult.Ok -> {
                        xs.add(item.value.first)
                        ys.add(item.value.second)
                    }

                    is CandleResult.Err -> {
                        nextBatch = item
                        return
                    }
                }
            } else {
                if (returnLastIncompleteBatch && xs.isNotEmpty() && ys.isNotEmpty()) {
                    break
                } else {
                    done = true
                    return
                }
            }
        }
        if (xs.isEmpty() || ys.isEmpty()) {
            done = true
        } else {
            nextBatch =
                try {
                    val xsStacked = Tensor.stack(xs, 0)
                    val ysStacked = Tensor.stack(ys, 0)
                    CandleResult.Ok(Pair(xsStacked, ysStacked))
                } catch (e: CandleException) {
                    CandleResult.Err(e)
                }
        }
    }

    override fun hasNext(): Boolean {
        prepareNext()
        return nextBatch != null
    }

    override fun next(): CandleResult<Pair<Tensor, Tensor>> {
        prepareNext()
        val batch = nextBatch ?: throw NoSuchElementException("No more elements in Batcher")
        nextBatch = null
        return batch
    }
}
