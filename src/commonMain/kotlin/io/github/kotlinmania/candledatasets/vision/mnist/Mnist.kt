// port-lint: source vision/mnist.rs
package io.github.kotlinmania.candledatasets.vision.mnist

/**
 * The MNIST hand-written digit dataset.
 *
 * The files can be obtained from the following link:
 * http://yann.lecun.com/exdb/mnist/
 */

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candledatasets.vision.Dataset
import io.github.kotlinmania.io.Source
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.readByteArray

public fun readU32(reader: Source): UInt = reader.readInt().toUInt()

public fun checkMagicNumber(reader: Source, expected: UInt) {
    val magicNumber = readU32(reader)
    if (magicNumber != expected) {
        bail("incorrect magic number $magicNumber != $expected")
    }
}

public fun readLabels(filename: Path): Tensor {
    val bufReader = SystemFileSystem.source(filename).buffered()
    checkMagicNumber(bufReader, 2049u)
    val samples = readU32(bufReader).toInt()
    val data = bufReader.readByteArray(samples)
    return Tensor.new(data.toUByteArray(), Shape(data.size), Device.Cpu)
}

public fun readLabels(filename: String): Tensor = readLabels(Path(filename))

public fun readImages(filename: Path): Tensor {
    val bufReader = SystemFileSystem.source(filename).buffered()
    checkMagicNumber(bufReader, 2051u)
    val samples = readU32(bufReader).toInt()
    val rows = readU32(bufReader).toInt()
    val cols = readU32(bufReader).toInt()
    val dataLen = samples * rows * cols
    val data = bufReader.readByteArray(dataLen)
    val tensor = Tensor.new(data.toUByteArray(), Shape(samples, rows * cols), Device.Cpu)
    return (tensor.toDType(DType.F32) / 255.0)
}

public fun readImages(filename: String): Tensor = readImages(Path(filename))

public fun loadDir(dir: Path): Dataset {
    val trainImages = readImages(Path(dir, "train-images-idx3-ubyte"))
    val trainLabels = readLabels(Path(dir, "train-labels-idx1-ubyte"))
    val testImages = readImages(Path(dir, "t10k-images-idx3-ubyte"))
    val testLabels = readLabels(Path(dir, "t10k-labels-idx1-ubyte"))
    return Dataset(
        trainImages = trainImages,
        trainLabels = trainLabels,
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

public fun loadMnistLike(
    datasetId: String,
    revision: String,
    testFilename: String,
    trainFilename: String,
): Dataset {
    // Blocked on hf-hub-kotlin (HFClientSync) and parquet-kotlin (SerializedFileReader)
    bail("loadMnistLike requires hf-hub and parquet, which are pending sibling ports")
}

public fun load(): Dataset =
    loadMnistLike(
        datasetId = "ylecun/mnist",
        revision = "refs/convert/parquet",
        testFilename = "mnist/test/0000.parquet",
        trainFilename = "mnist/train/0000.parquet",
    )
