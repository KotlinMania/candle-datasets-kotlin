// port-lint: source hub.rs
package io.github.kotlinmania.candledatasets

import io.github.kotlinmania.candle.bail

public sealed class Error(
    message: String,
) : Exception(message) {
    public data class HubError(
        val error: String,
    ) : Error("HubError : $error")

    public data class IoError(
        val error: String,
    ) : Error("IoError : $error")

    public data class ParquetError(
        val error: String,
    ) : Error("ParquetError : $error")
}

public const val PARQUET_REVISION: String = "refs/convert/parquet"

/**
 * Returns the path to filename in repo at revision. Downloads if not in cache.
 */
public fun cachedDownload(
    repo: Any,
    revision: String,
    filename: String,
): String {
    // Blocked on hf-hub-kotlin
    bail("cachedDownload requires hf-hub, which is a pending sibling port")
}

internal fun siblingToParquet(
    rfilename: String,
    repo: Any,
): Any {
    // Blocked on hf-hub-kotlin and parquet-kotlin
    bail("siblingToParquet requires hf-hub and parquet, which are pending sibling ports")
}

/**
 * Loads all .parquet files from a given dataset ID on the Hugging Face Hub.
 */
public fun fromHub(
    api: Any,
    datasetId: String,
): List<Any> {
    // Blocked on hf-hub-kotlin and parquet-kotlin
    bail("fromHub requires hf-hub and parquet, which are pending sibling ports")
}
