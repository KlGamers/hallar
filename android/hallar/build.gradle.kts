import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// AGP 9 compila Kotlin de forma nativa: no hace falta el plugin org.jetbrains.kotlin.android.
plugins {
    id("com.android.library")
}

val hostDir = layout.buildDirectory.dir("host-jni")

android {
    namespace = "io.github.klgamers.hallar"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static") }
        }
        consumerProguardFiles("consumer-rules.pro")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.all {
            it.dependsOn("hostJniBuild")
            it.systemProperty("java.library.path", hostDir.get().asFile.absolutePath)
            it.systemProperty("hallar.fixture", rootProject.file("../fixtures/tiny.hallar").absolutePath)
        }
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

val hostJniConfigure by tasks.registering(Exec::class) {
    val jdk = System.getenv("HALLAR_JNI_HOME") ?: System.getenv("JAVA_HOME") ?: ""
    commandLine(
        "cmake", "-S", "src/main/cpp", "-B", hostDir.get().asFile.absolutePath,
        "-DHALLAR_HOST=ON", "-DCMAKE_BUILD_TYPE=Release", "-DJAVA_HOME=$jdk",
    )
}

tasks.register<Exec>("hostJniBuild") {
    dependsOn(hostJniConfigure)
    commandLine("cmake", "--build", hostDir.get().asFile.absolutePath, "-j")
}
