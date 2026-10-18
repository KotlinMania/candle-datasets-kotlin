package io.github.kotlinmania.candledatasets.vision

import io.github.kotlinmania.candle.Tensor

public data class Dataset(
    public val trainImages: Tensor,
    public val trainLabels: Tensor,
    public val testImages: Tensor,
    public val testLabels: Tensor,
    public val labels: Int,
)
