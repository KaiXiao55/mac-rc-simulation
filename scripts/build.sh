#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${MACRC_BUILD_DIR:-$ROOT/.build}"
xcodebuild -project "$ROOT/macRCtrackerSimulator.xcodeproj" \
  -scheme macRCtrackerSimulator -configuration Debug -derivedDataPath "$BUILD_DIR" \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO build
