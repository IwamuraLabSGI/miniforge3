#!/bin/bash

set -ex

IFS="." read -a VER_ARR <<<"${PKG_VERSION}"

ARCH_FLAG=""
if [[ ${ARCH} == 64 ]]; then
    ARCH_FLAG="--enable-64bit"
fi

pushd tcl${PKG_VERSION}/unix
  # autoreconf -vfi
  ./configure  --prefix="${PREFIX}"  \
               --host=${HOST}        \
               ${ARCH_FLAG}
  make -j${CPU_COUNT} ${VERBOSE_AT}
  make install install-private-headers
popd

pushd tk${PKG_VERSION}/unix
  # autoreconf -vfi
  ./configure --prefix="${PREFIX}"        \
              --host=${HOST}              \
              --with-tcl="${PREFIX}"/lib  \
              --enable-aqua=yes           \
              ${ARCH_FLAG}
  make -j${CPU_COUNT} ${VERBOSE_AT}
  make install
popd

rm -rf "${PREFIX}"/{man,share}

# Link binaries to non-versioned names to make them easier to find and use.
ln -s "${PREFIX}"/bin/tclsh${VER_ARR[0]}.${VER_ARR[1]} "${PREFIX}"/bin/tclsh
ln -s "${PREFIX}"/bin/wish${VER_ARR[0]}.${VER_ARR[1]} "${PREFIX}"/bin/wish

# copy headers
cp "${SRC_DIR}"/tk${PKG_VERSION}/{unix,macosx,generic}/*.h "${PREFIX}"/include/

# Remove buildroot traces
sed -i.bak -e "s,${SRC_DIR}/tk${PKG_VERSION}/unix,${PREFIX}/lib,g" -e "s,${SRC_DIR}/tk${PKG_VERSION},${PREFIX}/include,g" ${PREFIX}/lib/tkConfig.sh
sed -i.bak -e "s,${SRC_DIR}/tcl${PKG_VERSION}/unix,${PREFIX}/lib,g" -e "s,${SRC_DIR}/tcl${PKG_VERSION},${PREFIX}/include,g" ${PREFIX}/lib/tclConfig.sh
rm -f ${PREFIX}/lib/tkConfig.sh.bak
rm -f ${PREFIX}/lib/tclConfig.sh.bak
