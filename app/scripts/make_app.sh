#!/bin/bash
# Build HydrogenSplat.app (release) from this checkout.
#   app/scripts/make_app.sh            -> app/build/HydrogenSplat.app
#   app/scripts/make_app.sh --install  -> also copies it to ~/Applications
# The app finds the engine through the repository path compiled into it (Setup can change it),
# so build it from the checkout you want it to drive.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
BIN="$(swift build -c release --show-bin-path)"
OUT="build/HydrogenSplat.app"
VERSION="0.1.$(git rev-list --count HEAD 2>/dev/null || echo 0)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN/HydrogenSplat" "$OUT/Contents/MacOS/HydrogenSplat"
cp -R "$BIN/HydrogenSplat_HydrogenSplat.bundle" "$OUT/Contents/Resources/"
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
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
</dict>
</plist>
PLIST
codesign --force --deep --sign - "$OUT" >/dev/null
echo "built $(pwd)/$OUT ($VERSION, $COMMIT)"
if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/HydrogenSplat.app"
  ditto "$OUT" "$HOME/Applications/HydrogenSplat.app"
  echo "installed $HOME/Applications/HydrogenSplat.app"
fi
