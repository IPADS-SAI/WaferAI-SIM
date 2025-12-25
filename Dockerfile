FROM ubuntu:22.04 AS build-systemc
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y g++ make unzip wget cmake
WORKDIR /opt
ENV SYSTEMC_VERSION=2.3.3
RUN wget https://gh-proxy.com/github.com/accellera-official/systemc/archive/refs/tags/2.3.3.zip && \
    unzip ${SYSTEMC_VERSION}.zip && cd systemc-${SYSTEMC_VERSION} && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/.. \
             -DCMAKE_INSTALL_LIBDIR=lib-linux64 \
             -DCMAKE_CXX_STANDARD=17 && \
    make -j$(nproc) && make install

FROM ubuntu:22.04
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make unzip wget git libcairo2-dev libsfml-dev xorg \
    software-properties-common gnupg ca-certificates
RUN wget https://gh-proxy.com/github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-linux-x86_64.sh -O /tmp/cmake.sh && \
    chmod +x /tmp/cmake.sh && \
    mkdir /opt/cmake && \
    /tmp/cmake.sh --skip-license --prefix=/opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    cmake --version

COPY --from=build-systemc /opt/systemc-2.3.3 /opt/systemc-2.3.3
ENV SYSTEMC_HOME=/opt/systemc-2.3.3
ENV LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib-linux64:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=${SYSTEMC_HOME}/include:$CPLUS_INCLUDE_PATH

WORKDIR /workspace
RUN git clone https://gh-proxy.com/github.com/doulujiyao12/npu-sim.git .
RUN mkdir build && cd build && \
    cmake .. -DSYSTEMC_HOME=${SYSTEMC_HOME} && cmake --build . -- -j$(nproc)

WORKDIR /workspace/build

CMD ["bash"]
