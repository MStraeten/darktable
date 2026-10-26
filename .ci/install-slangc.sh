#!/bin/bash

#    This file is part of darktable.
#    Copyright (C) 2026 darktable developers.
#
#    darktable is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    darktable is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with darktable.  If not, see <http://www.gnu.org/licenses/>.

# Installs a pinned release of the Slang compiler, which neither the Linux
# distributions, Homebrew nor MSYS2 package yet, and prints the path of
# slangc for -DSLANGC_EXECUTABLE. Only needed for builds with -DUSE_SLANG=ON;
# darktable needs no Slang library at run time.
#
# usage: install-slangc.sh <directory>

set -e

SLANG_VERSION=2026.20

# the MSYS2 runtime reports x86_64 on Windows on Arm too, so the CLANGARM64
# environment is recognized by its name
arch=$(uname -m)
[ "$MSYSTEM" = CLANGARM64 ] && arch=aarch64

case "$(uname -s)-$arch" in
  Linux-x86_64)
    asset=linux-x86_64.tar.gz
    sha256=e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9
    exe=slangc ;;
  Linux-aarch64)
    asset=linux-aarch64.tar.gz
    sha256=b0345743e2153d8c2bd33f56fb5696bcef63fc03513f054a1ae1b8d3dd0152a9
    exe=slangc ;;
  Darwin-arm64)
    asset=macos-aarch64.zip
    sha256=ede1865bf086442c8786c5b7c0470ce5f4fd4f032eefe881356726ff9c445f40
    exe=slangc ;;
  Darwin-x86_64)
    asset=macos-x86_64.zip
    sha256=a71d4a499d08b6178325a697a5a5c9fddc9eee5914daaf195451b283189e96ce
    exe=slangc ;;
  *_NT-*-x86_64)
    asset=windows-x86_64.zip
    sha256=e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174
    exe=slangc.exe ;;
  *_NT-*-aarch64)
    asset=windows-aarch64.zip
    sha256=4f18a7e76a59969bc9bf6ea596bd9e496a3c689af173990ae028b18f03e1ac1a
    exe=slangc.exe ;;
  *)
    echo "no Slang release for $(uname -s) $(uname -m)" >&2
    exit 1 ;;
esac

dir=$1
[ -n "$dir" ] || { echo "usage: $0 <directory>" >&2; exit 1; }
mkdir -p "$dir"
cd "$dir"

file=slang-$SLANG_VERSION-$asset
curl -sSLf --retry 3 -o "$file" \
  "https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/$file"

if command -v sha256sum > /dev/null; then
  echo "$sha256  $file" | sha256sum -c - >&2
else
  echo "$sha256  $file" | shasum -a 256 -c - >&2
fi

case "$file" in
  *.tar.gz) tar xzf "$file" ;;
  *.zip) unzip -q -o "$file" ;;
esac
rm "$file"

"$PWD/bin/$exe" -v >&2
echo "$PWD/bin/$exe"
