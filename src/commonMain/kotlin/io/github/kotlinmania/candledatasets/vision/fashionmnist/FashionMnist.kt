// port-lint: source vision/fashion_mnist.rs
package io.github.kotlinmania.candledatasets.vision.fashionmnist

import io.github.kotlinmania.candledatasets.vision.Dataset
import io.github.kotlinmania.candledatasets.vision.mnist.loadMnistLike

/**
 * Zalando Fashion MNIST dataset.
 * A slightly more difficult dataset that is drop-in compatible with MNIST.
 *
 * Taken from Hugging Face: zalando-datasets/fashion-mnist
 */

public fun load(): Dataset =
    loadMnistLike(
        datasetId = "zalando-datasets/fashion_mnist",
        revision = "refs/convert/parquet",
        testFilename = "fashion_mnist/test/0000.parquet",
        trainFilename = "fashion_mnist/train/0000.parquet",
    )
