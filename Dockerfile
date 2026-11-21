FROM rust:1.83-alpine AS builder
WORKDIR /build
ENV RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+crt-static"
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/shopping-sessions /shopping-sessions
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]