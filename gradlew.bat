@echo off
rem Gradle start up script for Windows

set GRADLE_VERSION=8.4
set DISTRIBUTION_URL=https://services.gradle.org/distributions/gradle-%GRADLE_VERSION%-bin.zip

set WRAPPER_DIR=%~dp0gradle\wrapper
set WRAPPER_JAR=%WRAPPER_DIR%\gradle-wrapper.jar
set WRAPPER_PROPERTIES=%WRAPPER_DIR%\gradle-wrapper.properties

if not exist "%WRAPPER_DIR%" mkdir "%WRAPPER_DIR%"

if not exist "%WRAPPER_PROPERTIES%" (
    echo distributionBase=GRADLE_USER_HOME > "%WRAPPER_PROPERTIES%"
    echo distributionPath=wrapper/dists >> "%WRAPPER_PROPERTIES%"
    echo distributionUrl=%DISTRIBUTION_URL% >> "%WRAPPER_PROPERTIES%"
    echo zipStoreBase=GRADLE_USER_HOME >> "%WRAPPER_PROPERTIES%"
    echo zipStorePath=wrapper/dists >> "%WRAPPER_PROPERTIES%"
)

java -jar "%WRAPPER_JAR%" %*
