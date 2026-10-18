// port-lint: source vision/cifar.rs
package io.github.kotlinmania.candledatasets.vision.cifar

/**
 * The CIFAR-10 dataset.
 *
 * The files can be downloaded from the following page:
 * https://www.cs.toronto.edu/~kriz/cifar.html
 * The binary version of the dataset is used.
 */

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candledatasets.vision.Dataset
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.readByteArray

public const val W: Int = 32
public const val H: Int = 32
public const val C: Int = 3
public const val BYTES_PER_IMAGE: Int = W * H * C + 1
public const val SAMPLES_PER_FILE: Int = 10000

public fun readFile(filename: Path): Pair<Tensor, Tensor> {
    val source = SystemFileSystem.source(filename).buffered()
    val data = source.readByteArray(SAMPLES_PER_FILE * BYTES_PER_IMAGE)
    val labels = UByteArray(SAMPLES_PER_FILE)
    val images = UByteArray(SAMPLES_PER_FILE * C * H * W)
    val imageSize = W * H * C
    val uData = data.asUByteArray()
    for (index in 0 until SAMPLES_PER_FILE) {
        val contentOffset = BYTES_PER_IMAGE * index
        labels[index] = uData[contentOffset]
        uData.copyInto(
            destination = images,
            destinationOffset = index * imageSize,
            startIndex = 1 + contentOffset,
            endIndex = contentOffset + BYTES_PER_IMAGE,
        )
    }
    val labelTensor = Tensor.new(labels, Shape(SAMPLES_PER_FILE), Device.Cpu)
    val imageTensor = Tensor.new(images, Shape(SAMPLES_PER_FILE, C, H, W), Device.Cpu)
    val normalizedImages = (imageTensor.toDtype(DType.F32) / 255.0)
    return Pair(normalizedImages, labelTensor)
}

public fun readFile(filename: String): Pair<Tensor, Tensor> = readFile(Path(filename))

public fun loadDir(dir: Path): Dataset {
    val (testImages, testLabels) = readFile(Path(dir, "test_batch.bin"))
    val trainNames = listOf(
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    )
    val trainPairs = trainNames.map { readFile(Path(dir, it)) }
    val trainImages = trainPairs.map { it.first }
    val trainLabels = trainPairs.map { it.second }
    return Dataset(
        trainImages = Tensor.cat(trainImages, 0),
        trainLabels = Tensor.cat(trainLabels, 0),
        testImages = testImages,
        testLabels = testLabels,
        labels = 10,
    )
}

public fun loadDir(dir: String): Dataset = loadDir(Path(dir))

public fun loadParquet(): Pair<Tensor, Tensor> {
    // Blocked on parquet-kotlin (SerializedFileReader) and image-kotlin (loadFromMemory)
    bail("loadParquet requires parquet and image, which are pending sibling ports")
}

public fun load(): Dataset {
    // Blocked on hf-hub-kotlin (HFClientSync) and parquet-kotlin (SerializedFileReader)
    bail("load requires hf-hub and parquet, which are pending sibling ports")
}
