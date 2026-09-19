#!/bin/bash
# Build HydrogenSplat.app (release) from this checkout.
#   app/scripts/make_app.sh            -> app/build/HydrogenSplat.app
#   app/scripts/make_app.sh --install  -> also copies it to ~/Applications
#   app/scripts/make_app.sh --spm      -> build with `swift build` instead of xcodebuild: everything
#                                         works except the splat viewer, which says why it is off
# The app finds the engine through the repository path compiled into it (Setup can change it),
# so build it from the checkout you want it to drive.
#
# Why xcodebuild: the splat viewer's renderer (MetalSplatter) loads its shaders from a compiled
# default.metallib inside its resource bundle. `swift build` copies .metal files without
# compiling them and writes no Info.plist into resource bundles; xcodebuild does both.
set -euo pipefail
cd "$(dirname "$0")/.."
INSTALL=0; SPM=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --spm) SPM=1 ;;
    *) echo "unknown option: $arg" >&2; exit 64 ;;
  esac
done

if [[ $SPM == 1 ]]; then
  swift build -c release
  BIN="$(swift build -c release --show-bin-path)"
else
  if ! xcodebuild -version >/dev/null 2>&1; then
    echo "xcodebuild needs the full Xcode, not just the command line tools:" >&2
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    echo "or build without the splat viewer:  app/scripts/make_app.sh --spm" >&2
    exit 1
  fi
  if ! xcrun -sdk macosx -f metal >/dev/null 2>&1; then
    echo "the Metal compiler is not installed (a separate download since Xcode 26):" >&2
    echo "  xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
  fi
  DD="build/xcode"
  xcodebuild -scheme HydrogenSplat -configuration Release -destination 'platform=macOS,arch=arm64' \
             -derivedDataPath "$DD" -quiet build
  BIN="$DD/Build/Products/Release"
fi
OUT="build/HydrogenSplat.app"
VERSION="0.1.$(git rev-list --count HEAD 2>/dev/null || echo 0)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN/HydrogenSplat" "$OUT/Contents/MacOS/HydrogenSplat"
# every SwiftPM resource bundle: the app's own and MetalSplatter's (its shaders). Contents/Resources
# is the first place the generated Bundle.module accessor looks inside an .app.
for b in "$BIN"/*.bundle; do
  [[ -e "$b" ]] && cp -R "$b" "$OUT/Contents/Resources/"
done
cp ../THIRD_PARTY_LICENSES.md "$OUT/Contents/Resources/THIRD_PARTY_LICENSES.md"
if [[ $SPM == 0 ]]; then
  # fail here, loudly, rather than in the viewer later
  if ! find "$OUT/Contents/Resources/MetalSplatter_MetalSplatter.bundle" -name default.metallib 2>/dev/null | grep -q .; then
    echo "xcodebuild produced no default.metallib in MetalSplatter_MetalSplatter.bundle — the splat viewer would be off." >&2
    echo "Products were: $(ls "$BIN" | tr '\n' ' ')" >&2
    exit 1
  fi
fi
cp Branding/AppIcon.icns "$OUT/Contents/Resources/AppIcon.icns"
cat > "$OUT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>HydrogenSplat</string>
  <key>CFBundleDisplayName</key><string>HydrogenSplat</string>
  <key>CFBundleIdentifier</key><string>com.sfouasnon.hydrogensplat</string>
  <key>CFBundleExecutable</key><string>HydrogenSplat</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${COMMIT}</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
</dict>
</plist>
PLIST
codesign --force --deep --sign - "$OUT" >/dev/null
echo "built $(pwd)/$OUT ($VERSION, $COMMIT)"
if [[ $INSTALL == 1 ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/HydrogenSplat.app"
  ditto "$OUT" "$HOME/Applications/HydrogenSplat.app"
  echo "installed $HOME/Applications/HydrogenSplat.app"
fi
