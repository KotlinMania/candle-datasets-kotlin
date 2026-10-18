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
    public fun batchSize(batchSize: Int): Batcher<I> {
        this.batchSize = batchSize
        return this
    }

    public fun returnLastIncompleteBatch(r: Boolean): Batcher<I> {
        this.returnLastIncompleteBatch = r
        return this
    }

    public companion object {
        public fun <I> new(inner: I): Batcher<I> = Batcher(inner)

        public fun <I : Iterator<Tensor>> new1(inner: I): Batcher1<I> =
            Batcher1(Iter1(inner))

        public fun <I : Iterator<Pair<Tensor, Tensor>>> new2(inner: I): Batcher2<I> =
            Batcher2(Iter2(inner))

        public fun <I : Iterator<CandleResult<Tensor>>> newR1(inner: I): BatcherResult1<I> =
            BatcherResult1(IterResult1(inner))

        public fun <I : Iterator<CandleResult<Pair<Tensor, Tensor>>>> newR2(inner: I): BatcherResult2<I> =
            BatcherResult2(IterResult2(inner))
    }
}

public class Iter1<I : Iterator<Tensor>>(
    public val inner: I,
)

public class Iter2<I : Iterator<Pair<Tensor, Tensor>>>(
    public val inner: I,
)

public class IterResult1<I : Iterator<CandleResult<Tensor>>>(
    public val inner: I,
)

public class IterResult2<I : Iterator<CandleResult<Pair<Tensor, Tensor>>>>(
    public val inner: I,
)

public class Batcher1<I : Iterator<Tensor>>(
    inner: Iter1<I>,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter1<I>>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>?> {

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

public class Batcher2<I : Iterator<Pair<Tensor, Tensor>>>(
    inner: Iter2<I>,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<Iter2<I>>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>?> {

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

public class BatcherResult1<I : Iterator<CandleResult<Tensor>>>(
    inner: IterResult1<I>,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult1<I>>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Tensor>?> {

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

public class BatcherResult2<I : Iterator<CandleResult<Pair<Tensor, Tensor>>>>(
    inner: IterResult2<I>,
    batchSize: Int = 16,
    returnLastIncompleteBatch: Boolean = false,
) : Batcher<IterResult2<I>>(inner, batchSize, returnLastIncompleteBatch),
    Iterator<CandleResult<Pair<Tensor, Tensor>>?> {

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
