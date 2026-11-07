#!/bin/sh
#
# Gradle start up script for UN*X
#
# This script downloads Gradle distribution if needed and executes it.

GRADLE_VERSION=8.4
DISTRIBUTION_URL="https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"

# Determine the wrapper directory
WRAPPER_DIR="$(dirname "$0")/gradle/wrapper"
WRAPPER_JAR="${WRAPPER_DIR}/gradle-wrapper.jar"
WRAPPER_PROPERTIES="${WRAPPER_DIR}/gradle-wrapper.properties"

# Create wrapper directory if needed
mkdir -p "${WRAPPER_DIR}"

# Create wrapper properties if not present
if [ ! -f "${WRAPPER_PROPERTIES}" ]; then
    cat > "${WRAPPER_PROPERTIES}" << PROPEOF
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=${DISTRIBUTION_URL}
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
PROPEOF
fi

# Execute Gradle wrapper
exec java -jar "${WRAPPER_JAR}" "$@"
