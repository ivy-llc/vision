FROM unifyai/ivy:latest-copsim

# Install Ivy
RUN rm -rf ivy && \
    git clone --recurse-submodules https://github.com/unifyai/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py develop --no-deps

# Install Ivy Demo Utils
RUN git clone --recurse-submodules https://github.com/unifyai/demo-utils && \
    cd demo-utils && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py develop --no-deps

# Install Ivy Mech
RUN git clone --recurse-submodules https://github.com/unifyai/mech && \
    cd mech && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py develop --no-deps

# Install Ivy Vision
RUN git clone --recurse-submodules https://github.com/unifyai/vision && \
    cd vision && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py develop --no-deps

COPY requirements.txt /
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

COPY ivy_vision_demos/requirements.txt /
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

# RUN python3 test_dependencies.py -fp requirements.txt,demo_requirements.txt && \
#    rm -rf requirements.txt && \
#    rm -rf demo_requirements.txt

WORKDIR /vision
