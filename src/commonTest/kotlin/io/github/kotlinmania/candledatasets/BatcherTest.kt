package io.github.kotlinmania.candledatasets

import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BatcherTest {

    @Test
    fun testBatcher1Exact() {
        val tensors = List(4) { i ->
            Tensor.new(floatArrayOf(i.toFloat(), (i + 1).toFloat()), Shape(2), Device.Cpu)
        }
        val batcher = Batcher.new1(tensors.iterator())
            .batchSize(2)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 2), batch1.value.shape().dims())

        val batch2 = batcher.next()
        assertNotNull(batch2)
        assertTrue(batch2 is CandleResult.Ok)
        assertEquals(listOf(2, 2), batch2.value.shape().dims())

        val batch3 = batcher.next()
        assertNull(batch3)
    }

    @Test
    fun testBatcher1IncompleteDiscarded() {
        val tensors = List(3) { i ->
            Tensor.new(floatArrayOf(i.toFloat()), Shape(1), Device.Cpu)
        }
        val batcher = Batcher.new1(tensors.iterator())
            .batchSize(2)
            .returnLastIncompleteBatch(false)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 1), batch1.value.shape().dims())

        val batch2 = batcher.next()
        assertNull(batch2)
    }

    @Test
    fun testBatcher1IncompleteRetained() {
        val tensors = List(3) { i ->
            Tensor.new(floatArrayOf(i.toFloat()), Shape(1), Device.Cpu)
        }
        val batcher = Batcher.new1(tensors.iterator())
            .batchSize(2)
            .returnLastIncompleteBatch(true)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 1), batch1.value.shape().dims())

        val batch2 = batcher.next()
        assertNotNull(batch2)
        assertTrue(batch2 is CandleResult.Ok)
        assertEquals(listOf(1, 1), batch2.value.shape().dims())

        val batch3 = batcher.next()
        assertNull(batch3)
    }

    @Test
    fun testBatcher2() {
        val pairs = List(4) { i ->
            val x = Tensor.new(floatArrayOf(i.toFloat()), Shape(1), Device.Cpu)
            val y = Tensor.new(floatArrayOf((i * 10).toFloat()), Shape(1), Device.Cpu)
            Pair(x, y)
        }
        val batcher = Batcher.new2(pairs.iterator())
            .batchSize(2)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 1), batch1.value.first.shape().dims())
        assertEquals(listOf(2, 1), batch1.value.second.shape().dims())

        val batch2 = batcher.next()
        assertNotNull(batch2)
        assertTrue(batch2 is CandleResult.Ok)

        val batch3 = batcher.next()
        assertNull(batch3)
    }

    @Test
    fun testBatcherResult1() {
        val results = List(4) { i ->
            CandleResult.Ok(Tensor.new(floatArrayOf(i.toFloat()), Shape(1), Device.Cpu))
        }
        val batcher = Batcher.newR1(results.iterator())
            .batchSize(2)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 1), batch1.value.shape().dims())
    }

    @Test
    fun testBatcherResult2() {
        val results = List(4) { i ->
            val x = Tensor.new(floatArrayOf(i.toFloat()), Shape(1), Device.Cpu)
            val y = Tensor.new(floatArrayOf((i * 2).toFloat()), Shape(1), Device.Cpu)
            CandleResult.Ok(Pair(x, y))
        }
        val batcher = Batcher.newR2(results.iterator())
            .batchSize(2)

        val batch1 = batcher.next()
        assertNotNull(batch1)
        assertTrue(batch1 is CandleResult.Ok)
        assertEquals(listOf(2, 1), batch1.value.first.shape().dims())
        assertEquals(listOf(2, 1), batch1.value.second.shape().dims())
    }
}
