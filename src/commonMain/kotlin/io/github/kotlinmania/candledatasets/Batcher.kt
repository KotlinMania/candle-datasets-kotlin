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

public class Iter1(
    public val inner: Iterator<Tensor>,
)

public class Iter2(
    public val inner: Iterator<Pair<Tensor, Tensor>>,
)

public class IterResult1(
    public val inner: Iterator<CandleResult<Tensor>>,
)

public class IterResult2(
    public val inner: Iterator<CandleResult<Pair<Tensor, Tensor>>>,
)

public class Batcher1(
    inner: Iter1,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter1>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>?> {

    override fun batchSize(batchSize: Int): Batcher1 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): Batcher1 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    override fun hasNext(): Boolean = inner.inner.hasNext()

    override fun next(): CandleResult<Tensor>? {
        val items = ArrayList<Tensor>(batchSize)
        for (i in 0 until batchSize) {
            if (inner.inner.hasNext()) {
                items.add(inner.inner.next())
            } else {
                if (returnLastIncompleteBatch && items.isNotEmpty()) {
                    break
                }
                return null
            }
        }
        return try {
            CandleResult.Ok(Tensor.stack(items, 0))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }
}

public class Batcher2(
    inner: Iter2,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter2>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>?> {

    override fun batchSize(batchSize: Int): Batcher2 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): Batcher2 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    override fun hasNext(): Boolean = inner.inner.hasNext()

    override fun next(): CandleResult<Pair<Tensor, Tensor>>? {
        val xs = ArrayList<Tensor>(batchSize)
        val ys = ArrayList<Tensor>(batchSize)
        for (i in 0 until batchSize) {
            if (inner.inner.hasNext()) {
                val pair = inner.inner.next()
                xs.add(pair.first)
                ys.add(pair.second)
            } else {
                if (returnLastIncompleteBatch && xs.isNotEmpty() && ys.isNotEmpty()) {
                    break
                }
                return null
            }
        }
        return try {
            val xsStacked = Tensor.stack(xs, 0)
            val ysStacked = Tensor.stack(ys, 0)
            CandleResult.Ok(Pair(xsStacked, ysStacked))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }
}

public class BatcherResult1(
    inner: IterResult1,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult1>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>?> {

    override fun batchSize(batchSize: Int): BatcherResult1 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): BatcherResult1 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    override fun hasNext(): Boolean = inner.inner.hasNext()

    override fun next(): CandleResult<Tensor>? {
        val items = ArrayList<CandleResult<Tensor>>(batchSize)
        for (i in 0 until batchSize) {
            if (inner.inner.hasNext()) {
                items.add(inner.inner.next())
            } else {
                if (returnLastIncompleteBatch && items.isNotEmpty()) {
                    break
                }
                return null
            }
        }
        val unwrapped = ArrayList<Tensor>(items.size)
        for (item in items) {
            when (item) {
                is CandleResult.Ok -> unwrapped.add(item.value)
                is CandleResult.Err -> return item
            }
        }
        return try {
            CandleResult.Ok(Tensor.stack(unwrapped, 0))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }
}

public class BatcherResult2(
    inner: IterResult2,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult2>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>?> {

    override fun batchSize(batchSize: Int): BatcherResult2 {
        super.batchSize(batchSize)
        return this
    }

    override fun returnLastIncompleteBatch(r: Boolean): BatcherResult2 {
        super.returnLastIncompleteBatch(r)
        return this
    }

    override fun hasNext(): Boolean = inner.inner.hasNext()

    override fun next(): CandleResult<Pair<Tensor, Tensor>>? {
        val xs = ArrayList<Tensor>(batchSize)
        val ys = ArrayList<Tensor>(batchSize)
        val errs = ArrayList<Throwable>()
        for (i in 0 until batchSize) {
            if (inner.inner.hasNext()) {
                when (val result = inner.inner.next()) {
                    is CandleResult.Ok -> {
                        xs.add(result.value.first)
                        ys.add(result.value.second)
                    }
                    is CandleResult.Err -> errs.add(result.error)
                }
            } else {
                if (returnLastIncompleteBatch && xs.isNotEmpty() && ys.isNotEmpty()) {
                    break
                }
                return null
            }
        }
        if (errs.isNotEmpty()) {
            return CandleResult.Err(errs.removeAt(0))
        }
        return try {
            val xsStacked = Tensor.stack(xs, 0)
            val ysStacked = Tensor.stack(ys, 0)
            CandleResult.Ok(Pair(xsStacked, ysStacked))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }
}
