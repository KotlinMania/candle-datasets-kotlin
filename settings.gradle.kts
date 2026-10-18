pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "candle-datasets-kotlin"

listOf(
    "../candle-kotlin",
    "../byteorder-kotlin",
    "../rand-kotlin",
    "../thiserror-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
